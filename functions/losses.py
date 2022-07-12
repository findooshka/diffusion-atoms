import torch
from jarvis.core.atoms import Atoms
import numpy as np

from functions.atomic import get_output, return_to_lattice, add_random_atoms, get_L_inv_estimate, B6_transform, B6_inv_transform, B9_to_B6, B6_to_B9, get_permutation
from functions.lattice import get_noise_mask, get_lattice_system, c_to_p_matrix
import logging


def lattice_loss(model,
                 atoms,
                 operations,
                 space_group,
                 sg_type,
                 t: torch.LongTensor,
                 e: torch.Tensor,
                 b: torch.Tensor,
                 device,
                 lengths_mult,
                 angles_mult,
                 gradient):
    ## gradient: function from basis6, edge vector to gradient of length wrt basis6
    lattice_system = get_lattice_system(space_group)
    noise_mask = get_noise_mask(lattice_system, device=device)
    x0 = torch.tensor(atoms.cart_coords, device=e.device)
    assert len(x0.shape) == 2
    #s_t = b.cumsum(dim=0).index_select(0, t)
    s_t = b.index_select(0, t)
    x = x0 + e * s_t.sqrt() * 0.1
    x_numpy = x.cpu().detach().numpy()
    
    x_numpy = return_to_lattice(x_numpy, c_to_p_matrix(sg_type, lattice_system=='monoclinic').detach().numpy()@atoms.lattice_mat)
    noised_atoms = Atoms(lattice_mat=atoms.lattice_mat,
                         coords=x_numpy,
                         elements=atoms.elements,
                         cartesian=True)
    
    lattice = torch.tensor(atoms.lattice_mat, device=device)
    lattice_neg = lattice.clone()
    lattice_neg[0] = -lattice_neg[0]
    lattice_neg = lattice_neg.detach()
    lattice6 = B9_to_B6(torch.tensor(atoms.lattice_mat, device=device), device=device)
    lattice6 = B6_transform(lattice6)
    lattice_noise = torch.normal(torch.zeros(noise_mask.shape[1], device=device))
    lattice_noise6 = noise_mask @ lattice_noise
    lattice_noise_scaled = torch.tensor(3*[lengths_mult] + 3*[angles_mult], device=device) * lattice_noise6 * s_t.sqrt()
    noised_lattice6 = lattice6 #+ lattice_noise_scaled
    noised_transformed_lattice6 = noised_lattice6.clone()
    noised_lattice6 = B6_inv_transform(noised_lattice6)
    noised_lattice9 = B6_to_B9(noised_lattice6, so=torch.linalg.det(lattice), device=device)
    if noised_lattice9 is None or torch.abs(torch.linalg.det(noised_lattice9)) < 1. * len(atoms.elements):
        return None
    noised_atoms2 = Atoms(lattice_mat=noised_lattice9.detach().cpu(),
                         coords=noised_atoms.frac_coords,
                         elements=atoms.elements,
                         cartesian=False)
    
    if get_permutation(lattice_system, noised_lattice9.detach().cpu(),) != (0, 1, 2):
        raise Exception("Lattice problem")
    
    L=torch.linalg.inv(noised_lattice9)@lattice
    #noise_estimate = get_output(noised_atoms2, operations, space_group, model, t, device, output_type="lattice", b6=noised_transformed_lattice6, gradient=gradient, emax=500)[1]
    noise_estimate = get_output(noised_atoms2, operations, space_group, sg_type, model, t, device, output_type="lattice", gradient=gradient, emax=500)[1]
    logging.info(f"noise: {lattice_noise}")
    logging.info(f"estimate: {noise_estimate}")
    return (noise_estimate - lattice_noise).square().mean()
    

def noise_estimation_loss(model,
                          atoms,
                          operations,
                          space_group,
                          sg_type,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          device,
                          gradient):
    x0 = torch.from_numpy(atoms.cart_coords).to(device=device)
    assert len(x0.shape) == 2
    
    s_t = b.index_select(0, t).view(-1, 1)
    x = x0 + e * s_t.sqrt()
    
    x_numpy = x.cpu().detach().numpy()
            
    x_numpy = return_to_lattice(x_numpy, atoms.lattice_mat)
    noised_atoms = Atoms(lattice_mat=atoms.lattice_mat,
                         coords=x_numpy,
                         elements=atoms.elements,
                         cartesian=True)

    pos_eps, lattice_eps = get_output(noised_atoms, operations, space_group, sg_type, model, t, device, output_type="edges", gradient=gradient)
    loss = (e - pos_eps).square().mean()
    
    return loss
    
