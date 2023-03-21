import torch
from jarvis.core.atoms import Atoms
import numpy as np

from functions.atomic import get_output, return_to_lattice, B6_transform, B6_inv_transform, B9_to_B6, B6_to_B9, get_permutation
from functions.lattice import get_noise_mask, get_lattice_system
import logging


def add_lattice_noise(s_t, space_group, atoms, device):
    lattice_system = get_lattice_system(space_group)
    noise_mask = get_noise_mask(lattice_system, device=device)
    
    lattice = torch.tensor(atoms.lattice_mat, device=device)
    lattice6 = B9_to_B6(torch.tensor(atoms.lattice_mat, device=device), device=device)
    lattice6 = B6_transform(lattice6)
    lattice_noise = torch.normal(torch.zeros(noise_mask.shape[1], device=device))
    lattice_noise6 = noise_mask @ lattice_noise
    lattice_noise_scaled = lattice_noise6 * s_t.sqrt()
    noised_lattice6 = lattice6 + lattice_noise_scaled
    noised_transformed_lattice6 = noised_lattice6.clone()
    noised_lattice6 = B6_inv_transform(noised_lattice6)
    noised_lattice9 = B6_to_B9(noised_lattice6, so=torch.linalg.det(lattice), device=device)
    if noised_lattice9 is None or torch.abs(torch.linalg.det(noised_lattice9)) < 1. * len(atoms.elements):
        return None
    if get_permutation(lattice_system, noised_lattice9.detach().cpu(),) != (0, 1, 2):
        raise Exception("Lattice problem")
    noised_atoms = Atoms(lattice_mat=noised_lattice9.detach().cpu(),
                         coords=atoms.frac_coords,
                         elements=atoms.elements,
                         cartesian=False)
    return noised_atoms, lattice_noise

def add_position_noise(s_t, space_group, sg_type, atoms, device):
    x0 = torch.tensor(atoms.cart_coords, device=device)
    e = torch.randn(*atoms.cart_coords.shape, device=device)
    x = x0 + e * s_t.sqrt()
    x_numpy = x.cpu().detach().numpy()
    x_numpy = return_to_lattice(x_numpy, atoms.lattice_mat, space_group, sg_type)
    noised_atoms = Atoms(lattice_mat=atoms.lattice_mat,
                         coords=x_numpy,
                         elements=atoms.elements,
                         cartesian=True)
    return noised_atoms, e
    
def lattice_loss(model,
                 atoms,
                 operations,
                 space_group,
                 sg_type,
                 t: torch.LongTensor,
                 b_pos: torch.Tensor,
                 b: torch.Tensor,
                 device,
                 gradient):
    s_t_pos = b_pos.index_select(0, t)
    noised_atoms, _ = add_position_noise(s_t_pos, space_group, sg_type, atoms, device)
    s_t = b.index_select(0, t)
    noised = add_lattice_noise(s_t, space_group, noised_atoms, device)
    logging.info(f"{s_t.item()}, {s_t_pos.item()}")
    if noised is None:
        return None
    noised_atoms, lattice_noise = noised
    noise_estimate = get_output(noised_atoms, operations, space_group, sg_type, model, t, device, output_type="lattice", gradient=gradient, emax=500)[1]
    if noise_estimate == None:
        return None
    logging.info(f"noise: {lattice_noise}")
    logging.info(f"estimate: {noise_estimate}")
    return (noise_estimate - lattice_noise).square().mean()
    

def noise_estimation_loss(model,
                          atoms,
                          operations,
                          space_group,
                          sg_type,
                          t: torch.LongTensor,
                          b: torch.Tensor,
                          device,
                          gradient):
    s_t = b.index_select(0, t)
    noised_atoms, e = add_position_noise(s_t, space_group, sg_type, atoms, device)
    pos_eps = get_output(noised_atoms, operations, space_group, sg_type, model, t, device, output_type="edges", gradient=gradient)[0]
    loss = (e - pos_eps).square().mean()
    
    return loss
    
