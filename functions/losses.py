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
                 device):
    lattice_noise_batch = []
    noised_atoms_batch = []
    for i in range(len(atoms)):
        s_t_pos = b_pos.index_select(0, t[i])
        noised_atoms, _ = add_position_noise(s_t_pos, space_group[i], sg_type[i], atoms[i], device)
        s_t = b.index_select(0, t[i])
        noised = add_lattice_noise(s_t, space_group[i], noised_atoms, device)
        if noised is None:
            return None
        noised_atoms_batch.append(noised[0])
        lattice_noise_batch.append(noised[1])
    noise_estimate = get_output(noised_atoms_batch, operations, space_group, sg_type, model, t, device, output_type="lattice", emax=-1)[1]
    if noise_estimate == None:
        return None
    #logging.info(f"noise: {lattice_noise}")
    #logging.info(f"estimate: {noise_estimate}")
    result = 0
    for i in range(len(atoms)):
        result = result + (noise_estimate[i] - lattice_noise_batch[i]).square().mean()
    result = result / len(atoms)
    return result
    

def noise_estimation_loss(model,
                          atoms,
                          operations,
                          space_group,
                          sg_type,
                          t: torch.LongTensor,
                          b: torch.Tensor,
                          device):
    noised_atoms = []
    e = []
    loss = 0
    for i in range(len(atoms)):
        s_t = b.index_select(0, t[i])
        noised = add_position_noise(s_t, space_group[i], sg_type[i], atoms[i], device)
        noised_atoms.append(noised[0])
        e.append(noised[1])
    pos_eps = get_output(noised_atoms, operations, space_group, sg_type, model, t, device, output_type="edges")[0]
    for i in range(len(atoms)):
        loss = loss + (e[i] - pos_eps[i]).square().mean()
    loss = loss / len(atoms)
    return loss
    
