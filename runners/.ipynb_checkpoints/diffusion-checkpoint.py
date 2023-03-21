import os
import logging
import time
import glob
import json

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import gc


#from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import noise_estimation_loss, lattice_loss, elements_loss
from datasets import get_dataset  #, data_transform, inverse_data_transform
from datasets.symmetries import get_operations, reduce_atoms
from functions.atomic import get_sampling_coords, add_random_atoms
from functions.lattice import get_lattice_system, p_to_c
from functions.denoising import langevin_dynamics
from jarvis.core.atoms import Atoms
from functions.symbolic_gradient import get_gradient_func
from models.dimenet_pp import DimeNetPP
from models.initializers import GlorotOrthogonal

import torchvision.utils as tvu


#def torch2hwcuint8(x, clip=False):
#    if clip:
#        x = torch.clamp(x, -1, 1)
#    x = (x + 1.0) / 2.0
#    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)


    if beta_schedule == "log":
        betas = (
            np.linspace(
                np.log(beta_start),
                np.log(beta_end),
                num_diffusion_timesteps,
                dtype=np.float64,
            )
        )
        betas = np.exp(betas)
    
    elif beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "sine":
        betas = np.linspace(
            0, 1, num_diffusion_timesteps, dtype=np.float64
        )
        betas = beta_end * (1 - np.cos(0.5 * np.pi * (beta_start+betas) / (beta_start+1)))
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        
        lattice_betas = get_beta_schedule(
            beta_schedule=config.lattice_diffusion.beta_schedule,
            beta_start=config.lattice_diffusion.beta_start,
            beta_end=config.lattice_diffusion.beta_end,
            num_diffusion_timesteps=config.lattice_diffusion.num_diffusion_timesteps,
        )
        self.lattice_betas = torch.from_numpy(lattice_betas).float().to(self.device)
        self.lattice_num_timesteps = config.lattice_diffusion.num_diffusion_timesteps
        
        self.gradient_func = get_gradient_func()
        self.element_loss_coef = np.load(config.data.elements_coef)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = DimeNetPP(emb_size=config.model.emb_size,
                          out_emb_size=config.model.out_emb_size,
                          int_emb_size=config.model.int_emb_size,
                          basis_emb_size=config.model.basis_emb_size,
                          num_blocks=config.model.num_blocks,
                          num_spherical=config.model.num_spherical,
                          num_radial=config.model.num_radial,
                          cutoff=config.model.cutoff,
                          envelope_exponent=config.model.envelope_exponent,
                          num_before_skip=config.model.num_before_skip,
                          num_after_skip=config.model.num_after_skip,
                          num_dense_output=config.model.num_dense_output,
                          extensive=config.model.extensive,
                          output_init=GlorotOrthogonal).to(self.device)
        
        
        loss_history = []
        #batch_loss = np.array([0., 0., 0.])
        batch_loss = np.array([0., 0.])

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        #if self.config.model.ema:
        #    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        #    ema_helper.register(model)
        #else:
        #    ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            loss_history = list(np.load(os.path.join(self.args.log_path, "loss.npy")))
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            #if self.config.model.ema:
            #    ema_helper.load_state_dict(states[4])

        bs = self.config.training.batch_size
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            time_start = time.time()
            iteration_time = 0
            for i, data_instance in enumerate(train_loader):
                atoms, operations, space_group, sg_type = data_instance['atoms'], data_instance['operations'], data_instance['space_group'], data_instance['sg_type']
                if get_lattice_system(space_group) == 'monoclinic' or get_lattice_system(space_group) == 'triclinic':
                    continue
                #if get_lattice_system(space_group) != 'monoclinic':
                #    continue
                if len(atoms.elements) * len(operations) > 500:
                    logging.warning("Skipping large structure")
                    continue
                #n = x.size(0)
                model.train()
                step += 1

                
                b = self.betas
                lattice_b = self.lattice_betas
                
                if step % bs == 1:
                    optimizer.zero_grad()
                    logging.info("zero_grad")
                
                t = torch.randint(
                    low=0, high=config.lattice_diffusion.num_diffusion_timesteps, size=(1,)
                ).to(self.device)
                loss = lattice_loss(model,
                                    atoms,
                                    operations,
                                    space_group,
                                    sg_type,
                                    t,
                                    b,
                                    lattice_b,
                                    self.device,
                                    gradient=self.gradient_func)
                if not (loss is None or torch.isnan(loss)):
                    loss *= self.config.training.lattice_loss_coef
                    tb_logger.add_scalar("lattice_loss", loss, global_step=step)
                    iteration_time = time.time() - time_start
                    logging.info(
                        "step: {:.3},\t loss (lattice): \t {:.3f}, \t time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), iteration_time, float(t))
                    )
                    (0.5 * loss / bs).backward()
                    batch_loss[1] += loss.item() / bs
                    del loss
                else:
                    logging.info("Bad lattice / noise, setting loss to 1, skipping the structure")
                    batch_loss[1] += 1. / bs
                    if loss is not None:
                        del loss
                    
                ############
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(1,)
                ).to(self.device)
                loss = noise_estimation_loss(model,
                                             atoms,
                                             operations,
                                             space_group,
                                             sg_type,
                                             t,
                                             b,
                                             self.device,
                                             gradient=self.gradient_func)
                tb_logger.add_scalar("loss", loss, global_step=step)
                iteration_time = time.time() - time_start
                logging.info(
                    "step: {:.3},\t loss (positions): \t {:.3f}, \t time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), iteration_time, float(t))
                )
                (0.5 * loss / bs).backward()
                batch_loss[0] += loss.item() / bs
                del loss

                ###########
                #t = torch.randint(
                #    low=0, high=self.num_timesteps, size=(1,)
                #).to(self.device)
                #loss = elements_loss(model,
                #                     atoms,
                #                     operations,
                #                     space_group,
                #                     sg_type,
                #                     t,
                #                     b,
                #                     lattice_b,
                #                     self.element_loss_coef,
                #                     self.device)
                #if not (loss is None or torch.isnan(loss)):
                #    tb_logger.add_scalar("loss", loss, global_step=step)
                #    iteration_time = time.time() - time_start
                #    logging.info(
                #        "step: {:.3},\t loss (elements): \t {:.6f}, \t time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), iteration_time, float(t))
                #    )

                #    (0.33 * loss / bs).backward()
                #    batch_loss[2] += loss.item() / bs
                #    del loss
                #else:
                #    logging.info("Bad lattice / noise, setting loss to 1, skipping the structure")
                #    batch_loss[2] += 1. / bs
                
                if step % bs == 0:
                    loss_history.append(batch_loss)
                    np.save(os.path.join(self.args.log_path, "loss"), loss_history)
                    loss_100_average = np.mean(loss_history[max(0, len(loss_history)-100):], axis=0)
                    logging.info("batch_loss: {}, loss_100_average: {}".format(batch_loss, loss_100_average))
                    batch_loss = [0., 0.]
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()

               
                if step % bs == 0 and (step // bs) % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step
                    ]
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step // bs)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                time_start = time.time()

    def sample(self):
        config = self.config
        model = DimeNetPP(emb_size=config.model.emb_size,
                          out_emb_size=config.model.out_emb_size,
                          int_emb_size=config.model.int_emb_size,
                          basis_emb_size=config.model.basis_emb_size,
                          num_blocks=config.model.num_blocks,
                          num_spherical=config.model.num_spherical,
                          num_radial=config.model.num_radial,
                          cutoff=config.model.cutoff,
                          envelope_exponent=config.model.envelope_exponent,
                          num_before_skip=config.model.num_before_skip,
                          num_after_skip=config.model.num_after_skip,
                          num_dense_output=config.model.num_dense_output,
                          extensive=config.model.extensive,
                          output_init=GlorotOrthogonal).to(self.device)

        #if getattr(self.config.sampling, "ckpt_id", None) is None:
        states = torch.load(
            os.path.join(self.args.log_path, "ckpt.pth"),
            map_location=self.config.device,
        )
        #else:
        #    states = torch.load(
        #       os.path.join(
        #            self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
        #        ),
        #        map_location=self.config.device,
        #    )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        model.eval()

        for order in config.sampling_order:
            for i in range(1, order.count+1):
                self.sample_sequence(model, order, i)

    def sample_sequence(self, model, order, index=None):
        config = self.config
        folder = os.path.join(order.image_folder, str(index))
        space_group = order.space_group
        operations, _, sg_type = get_operations(os.path.join(self.config.data.space_groups, str(space_group)))
        lattice_system = get_lattice_system(space_group)
        if hasattr(order, 'template'):
            atoms = Atoms.from_cif(order.template, use_cif2cell=False, get_primitive_atoms=False)
            atoms, _ = reduce_atoms(atoms, operations)
            atoms = p_to_c(atoms, sg_type, lattice_system)
        else:
            composition = []
            for element in order.composition:
                composition += order.composition[element] * [element]
            atoms = Atoms(coords = np.ones((len(composition), 3)),
                          lattice_mat = np.eye(3),
                          elements = composition,
                          cartesian=True)
    
        atoms.write_cif(os.path.join(folder, f"0_original.cif"), with_spg_info=False)
        

        with torch.no_grad():
            atoms_seq, x = self.sample_image(atoms,
                                             operations,
                                             sg_type,
                                             space_group,
                                             model, last=False,
                                             T=order.T,
                                             random_positions=order.random_positions,
                                             random_lattice=order.random_lattice,)

        #x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(atoms_seq)):
            atoms_seq[i].write_cif(os.path.join(folder, f"{i}.cif"), with_spg_info=False)
        atoms_seq[-1].write_cif(os.path.join(order.finals_dir, f"{index}.cif"), with_spg_info=False)
            #for j in range(x[i].size(0)):
                #tvu.save_image(
                #    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                #)


    def sample_image(self, atoms, operations, sg_type, space_group, model, random_positions, random_lattice, T, last=True):
        x = langevin_dynamics(atoms,
                              operations,
                              sg_type,
                              space_group,
                              self.betas,
                              self.lattice_betas,
                              model,
                              self.device,
                              T=T,
                              gradient=self.gradient_func,
                              noise_original_positions=self.args.noise_original,
                              random_positions=random_positions,
                              random_lattice=random_lattice)
        if last:
            x = x[0][-1]
        return x

    def test(self):
        args, config = self.args, self.config
        _, test_dataset = get_dataset(args, config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        model = DimeNetPP(emb_size=config.model.emb_size,
                          out_emb_size=config.model.out_emb_size,
                          int_emb_size=config.model.int_emb_size,
                          basis_emb_size=config.model.basis_emb_size,
                          num_blocks=config.model.num_blocks,
                          num_spherical=config.model.num_spherical,
                          num_radial=config.model.num_radial,
                          cutoff=config.model.cutoff,
                          envelope_exponent=config.model.envelope_exponent,
                          num_before_skip=config.model.num_before_skip,
                          num_after_skip=config.model.num_after_skip,
                          num_dense_output=config.model.num_dense_output,
                          extensive=config.model.extensive,
                          output_init=GlorotOrthogonal).to(self.device)
        model = model.to(self.device)
        
        model = torch.nn.DataParallel(model)
        
        results = {}
        
        
        for name in os.listdir(self.args.log_path):
            if name.find("000.pth") < 0:
                continue
            states = torch.load(os.path.join(self.args.log_path, name))
            print(name)
            model.load_state_dict(states[0])

            model.eval()
            loss = ([], [])
            
            for atoms in tqdm.tqdm(test_loader):
                for j in range(config.test.noise_rolls):
                    with torch.no_grad():
                        e = torch.randn(*atoms.cart_coords.shape, device=self.device)
                        b = self.betas

                        t = torch.randint(
                            low=config.test.timestep_min, high=config.test.timestep_max+1, size=(1,)
                        ).to(self.device)
                
                        #                                    atoms,
                        #                                    t,
                        #                                    e,
                        #                                    b,
                        #                                    self.device,
                        #                                    False,
                        #                                    0.,
                        #                                    False,
                        #                                    1.,
                        #                                    0.,
                        #                                    0.))
                        to_add = (   
                                     noise_estimation_loss(model,
                                                           atoms,
                                                           t,
                                                           e,
                                                           b,
                                                           self.device),
                                     lattice_loss(model, atoms, t, e, b,
                                                  self.device,
                                                  lengths_mult=self.config.lattice_diffusion.lengths_mult,
                                                  angles_mult=self.config.lattice_diffusion.angles_mult,
                                                  gradient=self.gradient_func),
                                 )
                        loss[0].append(to_add[0].item())
                        if to_add[1] is not None:
                            loss[1].append(to_add[1].item())
            results[name] = (np.mean(loss[0]), np.mean(loss[1]))
        with open(os.path.join(self.args.log_path, "test_loss_t_from_" + str(config.test.timestep_min) + "_to_" + str(config.test.timestep_max) + "_rolls_" + str(config.test.noise_rolls) + ".txt"), 'w') as f:
            json.dump(results, f)
        
