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
    result = {'atoms': [], 'L9': [], 'cart_coords': []}
    lattice = [B6_transform(B9_to_B6(torch.tensor(atoms[i].lattice_mat)))[mask_indices[i]].numpy() for i in range(len(atoms))]
    score = None
    while score is None:
        score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='lattice', emax=-1)[1]
    score = [score[i].cpu().numpy() / sigma.item()**0.5 for i in range(len(atoms))]
    for i in range(len(atoms)):
        z = np.random.normal(size=lattice[i].shape)
        lattice[i] = lattice[i] - 0.5*step*score[i] + step**0.5*noise_mult*z
        lattice[i] = expand(lattice[i], get_lattice_system(space_group[i]))
        lattice9 = B6_to_B9(B6_inv_transform(lattice[i]), 'cpu', np.linalg.det(atoms[i].lattice_mat)).cpu().detach().numpy()
        randomized_atoms = Atoms(coords = atoms[i].frac_coords,
                                 lattice_mat = lattice9,
                                 elements = atoms[i].elements,
                                 cartesian = False)
        result['atoms'].append(randomized_atoms)
        result['L9'].append(lattice9)
        result['cart_coords'].append(randomized_atoms.cart_coords)
    return result['atoms'], result['L9'], result['cart_coords']

def positions_step(xt, atoms, operations, space_group, sg_type, sigma, step, t, cp_matrix, model, device):
    score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='edges')[0]
    score = [score[i] / sigma**0.5 for i in range(len(atoms))]
    score = [score[i].cpu().detach().numpy() for i in range(len(atoms))]
    #print(f"Pos: Sigma: {sigma}, Noise_mult: {step**0.5}, Score_mult: {0.5*step/sigma**0.5}")
    result = {'atoms': [], 'cart_coords': []}
    for i in range(len(atoms)):
        z = np.random.normal(size=xt[i].shape)
        xt[i] = xt[i] - 0.5*step*score[i] + step**0.5 * z
        xt[i] = return_to_lattice(xt[i], cp_matrix[i]@atoms[i].lattice_mat, space_group[i], sg_type[i])
        randomized_atoms = Atoms(coords = xt[i],
                                 lattice_mat = atoms[i].lattice_mat,
                                 elements = atoms[i].elements,
                                 cartesian = True)
        result['atoms'].append(randomized_atoms)
        result['cart_coords'].append(xt[i])
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
                      T,
                      epsilon=1e-3,
                      lattice_epsilon=1e-2,
                      noise_original_positions=False):
    bs = len(atoms)
    lattice_system = [get_lattice_system(sg) for sg in space_group]
    cp_matrix = [c_to_p_matrix(sg_type[i], lattice_system[i]).numpy() for i in range(bs)]
    mask_indices = [get_mask(lattice_system[i]) for i in range(bs)]
    mask = [get_noise_mask(lattice_system[i]).numpy() for i in range(bs)]
    lattice = [expand(2*np.random.normal(size=len(mask_indices[i])), lattice_system[i]) for i in range(bs)]
    lattice = [B6_to_B9(B6_inv_transform(lattice[i])).numpy() for i in range(bs)]
    original_lattice = [atoms[i].lattice_mat for i in range(bs)]
    primitive_lattice = [cp_matrix[i]@lattice[i] for i in range(bs)]
    atoms = [Atoms(coords = atoms[i].cart_coords,
                   lattice_mat = lattice[i],
                   elements = atoms[i].elements,
                   cartesian = True)
             for i in range(bs)]
    xt = [get_sampling_coords(atoms[i], noise_original_positions, 0.15) for i in range(bs)]
    #else:
    #    xt_eps = [np.random.normal(size=atoms[i].cart_coords.shape) for i in range(bs)]
    #    xt = [atoms[i].cart_coords@np.linalg.inv(cp_matrix[i]@original_lattice[i])@primitive_lattice[i] for i in range(bs)]
        
    result = []
                
    max_T = np.max(T)
    for t in range(len(b)-1, -1, -1):
        sigma = b[t]
        step = epsilon * (sigma / b[0])
        step = step.cpu().detach().numpy()
        lattice_sigma = lattice_b[t]
        l_step = lattice_epsilon * (lattice_sigma / lattice_b[0])
        l_step = l_step.cpu().detach().numpy()
        
        for i in range(max_T):
            #print(i)
            #if not random_positions:
            #    xt = [xt_init[i] + xt_eps[i] * sigma[i].item()**0.5 for i in range(bs)]
            #    xt = [return_to_lattice(xt[i], lattice[i], space_group[i], sg_type[i]) for i in range(bs)]
            randomized_atoms = [Atoms(coords = xt[i],
                                      lattice_mat = lattice[i],
                                      elements = atoms[i].elements,
                                      cartesian = True)
                                for i in range(bs)]
            t_tensor = [torch.ones(1).to(device)*t for i in range(bs)]
            randomized_atoms, xt = positions_step(xt, randomized_atoms, operations, space_group, sg_type, sigma, step, t_tensor, cp_matrix, model, device)
            randomized_atoms = [Atoms(coords = xt[i],
                                    lattice_mat = lattice[i],
                                    elements = atoms[i].elements,
                                    cartesian = True)
                                for i in range(bs)]
            randomized_atoms, lattice, xt = lattice_step(randomized_atoms, operations, space_group, sg_type, mask_indices, mask, lattice_sigma, l_step, t_tensor, model, device, noise_mult=.3)
        output_atoms = [Atoms(coords = xt[i],
                              lattice_mat = cp_matrix[i]@lattice[i],
                              elements = atoms[i].elements,
                              cartesian = True)
                        for i in range(bs)]
        output_atoms = [apply_operations_atoms(output_atoms[i], operations[i], 4e-1)[0] for i in range(bs)]
        result.append([round_positions(operations[i], output_atoms[i]) for i in range(bs)])
    return result
   
