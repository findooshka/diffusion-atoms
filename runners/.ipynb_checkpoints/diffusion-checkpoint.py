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
from functions.losses import noise_estimation_loss, lattice_loss
from datasets import get_dataset  #, data_transform, inverse_data_transform
from functions.atomic import get_sampling_coords, add_random_atoms
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
        self.s_T = betas.sum()
        print("s_T: ", self.s_T)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.gradient_func = get_gradient_func()

        #alphas = 1.0 - betas
        #alphas_cumprod = alphas.cumprod(dim=0)
        #alphas_cumprod_prev = torch.cat(
        #    [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        #)
        #posterior_variance = (
        #    betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        #)
        #if self.model_var_type == "fixedlarge":
        #    self.logvar = betas.log()
        #elif self.model_var_type == "fixedsmall":
        #    self.logvar = posterior_variance.clamp(min=1e-20).log()

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
        
        
        loss_ema = [0]
        batch_loss = 0.

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

            loss_ema = list(np.load(os.path.join(self.args.log_path, "loss_ema.npy")))
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            #if self.config.model.ema:
            #    ema_helper.load_state_dict(states[4])

        bs = self.config.training.batch_size
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, atoms in enumerate(train_loader):
                #n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                e = torch.randn(*atoms.cart_coords.shape, device=self.device)
                b = self.betas

                # antithetic sampling
                #t = torch.randint(
                #    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                #).to(self.device)
                #t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(1,)
                ).to(self.device)
                
                loss_type = np.random.randint(2)
                

                #if False:#loss_type == 0:
                    #tb_logger.add_scalar("added_loss", loss, global_step=step)
                    #logging.info(
                    #    "step: {:.3},\t loss (atom count): \t {:.3f}, \t data time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), data_time / (i+1), float(t))
                    #)
                if loss_type == 1:
                    loss = lattice_loss(model,
                                        atoms,
                                        t,
                                        e,
                                        b,
                                        self.device,
                                        lengths_mult=self.config.lattice_diffusion.lengths_mult,
                                        angles_mult=self.config.lattice_diffusion.angles_mult,
                                        gradient=self.gradient_func)
                    if loss is None:
                        step -= 1
                        continue
                    loss *= self.config.training.lattice_loss_coef
                    tb_logger.add_scalar("lattice_loss", loss, global_step=step)
                    logging.info(
                        "step: {:.3},\t loss (lattice): \t {:.3f}, \t data time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), data_time / (i+1), float(t))
                    )
                else:
                    loss = noise_estimation_loss(model,
                                                 atoms,
                                                 t,
                                                 e,
                                                 b,
                                                 self.device)
                    tb_logger.add_scalar("loss", loss, global_step=step)
                    logging.info(
                        "step: {:.3},\t loss (positions): \t {:.3f}, \t data time: {:.6} \t timestep: {}".format(step/self.config.training.batch_size, loss.item(), data_time / (i+1), float(t))
                    )

                if step % bs == 1:
                    optimizer.zero_grad()
                    print("zero_grad")
                
                
                (loss / bs).backward()
                batch_loss += loss.item() / bs
                del loss
                
                #for obj in gc.get_objects():
                #    if torch.is_tensor(obj):
                #        print(type(obj), obj.size())
                
                if step % bs == 0:
                    loss_ema.append(loss_ema[-1] * 0.99 + batch_loss * 0.01)
                    logging.info("batch_loss: {}, loss_ema: {}".format(batch_loss, loss_ema[-1]))
                    batch_loss = 0.
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()

                #if self.config.model.ema:
                #    ema_helper.update(model)

                if step % bs == 0 and (step // bs) % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step
                    ]
                    #if self.config.model.ema:
                    #    states.append(ema_helper.state_dict())
                    print(os.path.join(self.args.log_path, "loss_ema"))
                    np.save(os.path.join(self.args.log_path, "loss_ema"), loss_ema)
                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step // bs)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

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

        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(
                os.path.join(self.args.log_path, "ckpt.pth"),
                map_location=self.config.device,
            )
        else:
            states = torch.load(
               os.path.join(
                    self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                ),
                map_location=self.config.device,
            )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        model.eval()

        if self.args.count == 1:
            self.sample_sequence(model)
        else:
            for i in range(1, self.args.count+1):
                self.sample_sequence(model, i)

    def sample_multiple(self, model):
        config = self.config

    def sample_sequence(self, model, index=None):
        config = self.config
        if index is not None:
            folder = os.path.join(self.args.image_folder, str(index))
        else:
            folder = self.args.image_folder
     
        composition = []
        for element in config.sampling.composition_dict:
            composition += config.sampling.composition_dict[element] * [element]
        atoms = Atoms(coords = 3*np.random.normal(size=(len(composition), 3)),
                      lattice_mat = 3*np.random.normal(size=(3,3)),
                      elements = composition,
                      cartesian=True)
        #atoms = Atoms.from_cif(self.config.sampling.sample_structure)
        #atoms = Atoms(coords = 3*np.random.normal(size=(17, 3)),
        #          lattice_mat = 3*np.random.normal(size=(3,3)),
        #          elements = ['H']*12 + ['Li']*2 + ['B']*3,
        #          cartesian=True)
    
        atoms.write_cif(os.path.join(folder, f"0_original.cif"), with_spg_info=False)
        if not config.sampling.n_atoms_constant:
            atoms, _ = add_random_atoms(atoms, config.sampling.n_atoms_lambda, scale_with_natoms=False)
        

        with torch.no_grad():
            atoms_seq, x = self.sample_image(atoms, model, last=False,
                                             remove_atoms=not config.sampling.n_atoms_constant,
                                             remove_atoms_mult=config.sampling.remove_atom_mult*config.sampling.n_atoms_lambda/config.diffusion.num_diffusion_timesteps,
                                             lattice_noise=config.sampling.lattice_noise,
                                             lattice_noise_mult=config.sampling.lattice_noise_mult,)

        #x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(atoms_seq)):
            atoms_seq[i].write_cif(os.path.join(folder, f"{i}.cif"), with_spg_info=False)
            #for j in range(x[i].size(0)):
                #tvu.save_image(
                #    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                #)

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, atoms, model, remove_atoms, remove_atoms_mult, lattice_noise, lattice_noise_mult, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                #skip = self.num_timesteps // self.args.timesteps
                #seq = range(0, self.num_timesteps, skip)
                skip = 1#self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps, langevin_dynamics

            x = langevin_dynamics(atoms, self.betas, model, self.device, gradient=self.gradient_func,
                                  lengths_mult=self.config.lattice_diffusion.lengths_mult,
                                  angles_mult=self.config.lattice_diffusion.angles_mult,
                                  noise_original_positions=self.args.noise_original)
            #x = generalized_steps(x, atoms, seq, model, self.betas, self.device,
            #                      eta=self.args.eta,
            #                      max_t=self.config.sampling.max_t,
            #                      remove_atoms=remove_atoms,
            #                      remove_atoms_mult=remove_atoms_mult,
            #                      lattice_noise=lattice_noise,
            #                      lattice_noise_mult=lattice_noise_mult)
        else:
            raise NotImplementedError
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
        
