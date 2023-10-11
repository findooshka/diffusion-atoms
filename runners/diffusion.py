import os
import logging
import time
import glob
import json

import numpy as np
import tqdm
import torch
import torch.utils.data as data
torch.multiprocessing.set_sharing_strategy('file_system')
import gc


from functions import get_optimizer
from functions.losses import noise_estimation_loss, lattice_loss
from datasets import get_dataset, get_batch
from datasets.symmetries import get_operations, reduce_atoms
from functions.atomic import get_sampling_coords
from functions.lattice import get_lattice_system, p_to_c
from functions.denoising import langevin_dynamics
from jarvis.core.atoms import Atoms
from models.dimenet_pp import DimeNetPP
from models.initializers import GlorotOrthogonal
from runners.ema import EMAHelper
import argparse

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
        
        
        self.model = DimeNetPP(emb_size=config.model.emb_size,
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
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = self.model
        
        
        loss_history = []

        
        optimizer = get_optimizer(self.config, model.parameters())
        ema_helper = EMAHelper(mu=self.config.training.ema_mu)
        ema_helper.register(model)

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            loss_history = list(np.load(os.path.join(self.args.log_path, "loss_saved.npy")))
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            ema_helper.load_state_dict(states[4])

        bs = self.config.training.batch_size
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            time_start = time.time()
            iteration_time = 0
            data_iterator = iter(train_loader)
            atoms, operations, space_group, sg_type = get_batch(data_iterator, bs)
            while len(atoms) > 0:
                current_batch_size = len(atoms)
                model.train()
                
                b = self.betas
                lattice_b = self.lattice_betas
                
                optimizer.zero_grad()
                
                t = torch.randint(
                    low=0, high=config.lattice_diffusion.num_diffusion_timesteps, size=(current_batch_size,)
                ).to(self.device)
                loss = lattice_loss(model,
                                    atoms,
                                    operations,
                                    space_group,
                                    sg_type,
                                    t,
                                    b,
                                    lattice_b,
                                    self.device)
                if not (loss is None or torch.isnan(loss)):
                    loss *= self.config.training.lattice_loss_coef
                    iteration_time = time.time() - time_start
                    logging.info(
                        "step: {},\t loss (lattice): \t {:.3f}, \t time: {:.6}".format(step, loss.item(), iteration_time)
                    )
                    (0.5 * loss).backward()
                    batch_lattice_loss = loss.item()
                else:
                    logging.warning("Failed to give lattice score estimate, skipping the structure")
                    batch_lattice_loss = .5
                if loss is not None:
                    del loss
                    
                ############
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(current_batch_size,)
                ).to(self.device)
                loss = noise_estimation_loss(model,
                                             atoms,
                                             operations,
                                             space_group,
                                             sg_type,
                                             t,
                                             b,
                                             self.device)
                if not (loss is None or torch.isnan(loss)):
                    iteration_time = time.time() - time_start
                    logging.info(
                        "step: {},\t loss (positions): \t {:.3f}, \t time: {:.6}".format(step, loss.item(), iteration_time)
                    )
                    (0.5 * loss).backward()
                    batch_position_loss = loss.item()
                else:
                    logging.warning("Failed to give position score estimate, skipping the structure")
                    batch_position_loss = .5
                if loss is not None:
                    del loss

                loss_history.append((batch_position_loss, batch_lattice_loss))
                np.save(os.path.join(self.args.log_path, "loss"), loss_history)
                loss_100_average = np.mean(loss_history[max(0, len(loss_history)-100):], axis=0)
                logging.info("batch_loss: {}, loss_100_average: {}".format((batch_position_loss, batch_lattice_loss), loss_100_average))
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()
                ema_helper.update(model)

               
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    np.save(os.path.join(self.args.log_path, "loss_saved"), loss_history)
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                        ema_helper.state_dict()
                    ]
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                atoms, operations, space_group, sg_type = get_batch(data_iterator, bs)
                time_start = time.time()
                step += 1
                
    def sampling_load_model(self, path):
        states = torch.load(
            path,
            map_location=self.config.device,
        )
        self.model.load_state_dict(states[0], strict=True)
        ema_helper = EMAHelper(mu=self.config.training.ema_mu)
        ema_helper.register(self.model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(self.model)
        self.model.eval()

    def sample(self):
        if self.args.timestamp != -1:
            ckpt_name = f"ckpt_{self.args.timestamp}.pth"
        else:
            ckpt_name = "ckpt.pth"
        self.sampling_load_model(os.path.join(self.args.log_path, ckpt_name))
        order_batch, index_batch = [], []
        for order in self.config.sampling_order:
            logging.info(f"Current space group: {order.space_group}")
            for i in tqdm.tqdm(range(1, order.count+1)):
                order_batch.append(order)
                index_batch.append(i)
                if len(order_batch) == self.config.sampling.batch_size:
                    self.sample_sequence(self.model, order_batch, index_batch)
                    order_batch, index_batch = [], []
        if len(order_batch) > 0:
            self.sample_sequence(self.model, order_batch, index_batch)
            order_batch, index_batch = [], []
    
    def sample_sequence(self, model, order, index, final_file_name=None):
        config = self.config
        bs = len(order)
        space_group = [order[i].space_group for i in range(bs)]
        lattice_system = [get_lattice_system(space_group[i]) for i in range(bs)]
        operations, sg_type, atoms = [], [], []
        for i in range(bs):
            loaded_operations = get_operations(os.path.join(self.config.data.space_groups, str(space_group[i])))
            operations.append(loaded_operations[0])
            sg_type.append(loaded_operations[2])
            if hasattr(order[i], 'template'):
                atoms.append(Atoms.from_cif(order[i].template, use_cif2cell=False, get_primitive_atoms=False))
                atoms[i] = reduce_atoms(atoms[i], operations[i], check_consistency=True)
                atoms[i] = p_to_c(atoms[i], sg_type[i], lattice_system[i])
                atoms[i] = Atoms(coords=atoms[i].cart_coords+np.random.normal(scale=0.01, size=atoms[i].cart_coords.shape),
                                 lattice_mat=atoms[i].lattice_mat,
                                 cartesian=True,
                                 elements=atoms[i].elements)
            else:
                composition = []
                for element in order[i].composition:
                    composition += order[i].composition[element] * [element]
                atoms.append(Atoms(coords = np.ones((len(composition), 3)),
                                   lattice_mat = np.eye(3),
                                   elements = composition,
                                   cartesian=True))
            if not order[i].only_final:
                folder = os.path.join(order[i].image_folder, str(index[i]))
                atoms.write_cif(os.path.join(folder, f"0_original.cif"), with_spg_info=False)
        

        with torch.no_grad():
            atoms_seq = langevin_dynamics(atoms,
                                          operations,
                                          sg_type,
                                          space_group,
                                          self.betas,
                                          self.lattice_betas,
                                          model,
                                          self.device,
                                          T=[order[i].T for i in range(bs)],
                                          noise_original_positions=self.args.noise_original)
        for i in range(bs):
            if not order[i].only_final:
                for j in range(len(atoms_seq)):
                    atoms_seq[j][i].write_cif(os.path.join(folder, f"{j}.cif"), with_spg_info=False)
            atoms_seq[-1][i].write_cif(os.path.join(order[i].finals_dir, f"{index[i] if final_file_name is None else final_file_name}.cif"), with_spg_info=False)

    @torch.no_grad()
    def test(self):
        # calculate the learning curve on test data
        args, config = self.args, self.config
        _, test_dataset = get_dataset(args, config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=None,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        bs = self.config.training.batch_size
        
        results = {}
        
        
        for name in tqdm.tqdm(os.listdir(self.args.log_path)):
            if len(name) < 9 or name[-7:] != '000.pth' or int(name[-8]) % 5 == 0:
                continue
            logging.info(f"Calculating test loss for: {name}")
            logging.info(name)
            self.sampling_load_model(os.path.join(self.args.log_path, name))
            loss = ([], [])
            data_iterator = iter(test_loader)
            atoms, operations, space_group, sg_type = get_batch(data_iterator, bs)
            while len(atoms) > 0:
                #atoms, operations, space_group, sg_type = data_instance['atoms'], data_instance['operations'], data_instance['space_group'], data_instance['sg_type']
                #if get_lattice_system(space_group) == 'monoclinic' or get_lattice_system(space_group) == 'triclinic':
                #    continue
                for j in range(config.test.noise_rolls):
                    b = self.betas
                    lattice_b = self.lattice_betas

                    t = torch.randint(
                        low=config.test.timestep_min, high=config.test.timestep_max+1, size=(len(atoms),)
                    ).to(self.device)
                    to_add = (   
                                 noise_estimation_loss(self.model,
                                                       atoms,
                                                       operations,
                                                       space_group,
                                                       sg_type,
                                                       t,
                                                       b,
                                                       self.device),
                                 lattice_loss(self.model,
                                              atoms,
                                              operations,
                                              space_group,
                                              sg_type,
                                              t,
                                              b,
                                              lattice_b,
                                              self.device),
                    )
                    loss[0].append(to_add[0].item())
                    if to_add[1] is not None:
                        loss[1].append(to_add[1].item())
                atoms, operations, space_group, sg_type = get_batch(data_iterator, bs)
            results[name] = (np.mean(loss[0]), np.mean(loss[1]))
        with open(os.path.join(self.args.log_path, "test_loss_t_from_" + str(config.test.timestep_min) + "_to_" + str(config.test.timestep_max) + "_rolls_" + str(config.test.noise_rolls) + ".txt"), 'w') as f:
            json.dump(results, f)
