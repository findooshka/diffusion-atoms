import torch
from functions.atomic import return_to_lattice, get_output, B6_transform, B6_inv_transform, get_sampling_coords, B9_to_B6, B6_to_B9
from functions.lattice import sample_lattice, c_to_p_matrix, get_lattice_system, get_mask, get_noise_mask, expand
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
import numpy as np
from collections import Counter
from datasets.symmetries import reduce_atoms, apply_operations_atoms
from postprocessing.round_positions import round_positions

def compute_st(b, t):
    return b.cumsum(dim=0).index_select(0, t.long()).view(-1, 1)
    
def lattice_step(atoms, operations, space_group, sg_type, mask_indices, mask, sigma, step, t, model, device, noise_mult=.15):
    lattice = B6_transform(B9_to_B6(torch.tensor(atoms.lattice_mat)))[mask_indices].numpy()
    lattice_system = get_lattice_system(space_group)
    score = None
    while score is None:
        score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='lattice', emax=-1)[1].cpu().numpy()
    score /= sigma.item()**0.5
    z = np.random.normal(size=lattice.shape)
    #print(f"Lattice: Sigma: {sigma}, Lattice: {lattice}, Noise_mult: {noise_mult*step**0.5}, Score_mult: {0.5*step/sigma**0.5}")
    lattice = lattice - 0.5*step*score + step**0.5*noise_mult*z
    lattice = expand(lattice, lattice_system)
    lattice9 = B6_to_B9(B6_inv_transform(lattice), 'cpu', np.linalg.det(atoms.lattice_mat)).cpu().detach().numpy()
    randomized_atoms = Atoms(coords = atoms.frac_coords,
                             lattice_mat = lattice9,
                             elements = atoms.elements,
                             cartesian = False)
    return randomized_atoms, lattice9, randomized_atoms.cart_coords

def positions_step(xt, atoms, operations, space_group, sg_type, sigma, step, t, cp_matrix, model, device):
    z = np.random.normal(size=xt.shape)
    score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='edges')[0] / sigma**0.5        
    score = score.cpu().detach().numpy()
    #print(f"Pos: Sigma: {sigma}, Noise_mult: {step**0.5}, Score_mult: {0.5*step/sigma**0.5}")
    xt = xt - 0.5*step*score + step**0.5 * z
    xt = return_to_lattice(xt, cp_matrix@atoms.lattice_mat, space_group, sg_type)
    atoms = Atoms(coords = xt,
                  lattice_mat = atoms.lattice_mat,
                  elements = atoms.elements,
                  cartesian = True)
    return atoms, xt

@torch.no_grad()
def langevin_dynamics(atoms,
                      operations,
                      sg_type,
                      space_group,
                      b,
                      lattice_b,
                      model,
                      device,
                      T=1,
                      epsilon=1e-3,
                      lattice_epsilon=1e-2,
                      noise_original_positions=False,
                      random_positions=True,
                      random_lattice=True):
    lattice_system = get_lattice_system(space_group)
    cp_matrix = c_to_p_matrix(sg_type, lattice_system).numpy()
    mask_indices = get_mask(lattice_system)
    mask = get_noise_mask(lattice_system).numpy()
    if random_lattice:
        #lattice = B6_to_B9(B6_inv_transform(sample_lattice(space_group, atoms, len(operations)))).numpy()
        lattice = expand(2*np.random.normal(size=len(mask_indices)), lattice_system)
        lattice = B6_to_B9(B6_inv_transform(lattice)).numpy()
        original_lattice = atoms.lattice_mat
    else:
        lattice = atoms.lattice_mat
    primitive_lattice = cp_matrix@lattice
    atoms = Atoms(coords = atoms.cart_coords,
                  lattice_mat = lattice,
                  elements = atoms.elements,
                  cartesian = True)
    if random_positions:
        xt = get_sampling_coords(atoms, noise_original_positions, 0.15)
    else:
        xt_eps = np.random.normal(size=atoms.cart_coords.shape)
        if random_lattice:
            xt = atoms.cart_coords@np.linalg.inv(cp_matrix@original_lattice)@primitive_lattice
        else:
            xt = atoms.cart_coords
        
    result = []
                
    
    for t in range(len(b)-1, -1, -1):
        sigma = b[t]
        step = epsilon * (sigma / b[0])
        step = step.cpu().detach().numpy()
        lattice_sigma = lattice_b[t]
        l_step = lattice_epsilon * (lattice_sigma / lattice_b[0])
        l_step = l_step.cpu().detach().numpy()
        
        
        for i in range(T):
            if not random_positions:
                xt = xt_init + xt_eps * sigma.item()**0.5
                xt = return_to_lattice(xt, lattice, space_group, sg_type)
            randomized_atoms = Atoms(coords = xt,
                                     lattice_mat = lattice,
                                     elements = atoms.elements,
                                     cartesian = True)
            t_tensor = torch.ones(1).to(device)*t
            if random_positions:
                randomized_atoms, xt = positions_step(xt, randomized_atoms, operations, space_group, sg_type, sigma, step, t_tensor, cp_matrix, model, device)
            if random_lattice:
                randomized_atoms = Atoms(coords = xt,
                                     lattice_mat = lattice,
                                     elements = atoms.elements,
                                     cartesian = True)
                randomized_atoms, lattice, xt = lattice_step(randomized_atoms, operations, space_group, sg_type, mask_indices, mask, lattice_sigma, l_step, t_tensor, model, device, noise_mult=.3)
        output_atoms = Atoms(coords = xt,
                             lattice_mat = cp_matrix@lattice,
                             elements = atoms.elements,
                             cartesian = True)
        output_atoms = apply_operations_atoms(output_atoms, operations, 4e-1)[0]
        result.append(round_positions(operations, output_atoms))
    return result
   
