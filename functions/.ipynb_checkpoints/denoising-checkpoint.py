import torch
from functions.atomic import return_to_lattice, get_output, B6_transform, B6_inv_transform, get_sampling_coords, B9_to_B6, B6_to_B9
from functions.lattice import sample_lattice, c_to_p_matrix, get_lattice_system, get_mask, get_noise_mask, expand
from jarvis.core.atoms import Atoms
from jarvis.core.specie import atomic_numbers_to_symbols
import numpy as np
from collections import Counter
from datasets.symmetries import reduce_atoms, apply_operations_atoms

def compute_st(b, t):
    return b.cumsum(dim=0).index_select(0, t.long()).view(-1, 1)

def remove_atom(atoms, fake_probabilities, mult, softmax_l=20.):
    fake_softmax = np.exp(softmax_l*fake_probabilities)
    fake_softmax = np.ravel(fake_softmax) / fake_softmax.sum()
    #print(fake_softmax)
    remove_i = np.argmax(np.random.multinomial(1, fake_softmax))
    #remove_i = np.argmax(fake_probabilities)
    if len(atoms.elements) > 1 and np.random.rand() < mult * fake_probabilities[remove_i]:
        x = atoms.cart_coords
        x = np.delete(x, remove_i, 0)
        elements = atoms.elements.copy()
        print("removed", elements.pop(remove_i))
        return Atoms(coords = x,
                     lattice_mat = atoms.lattice_mat,
                     elements = elements,
                     cartesian = True)
    return atoms
    
def lattice_step(atoms, operations, space_group, sg_type, mask_indices, mask, sigma, step, t, model, device, gradient):
    lattice = B6_transform(B9_to_B6(torch.tensor(atoms.lattice_mat)))[mask_indices].numpy()
    lattice_system = get_lattice_system(space_group)
    score = None
    while score is None:
        score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='lattice', gradient=gradient, emax=200)[1].cpu().numpy()
    print(score, sigma, step)
    score /= sigma.item()**0.5
    #score[:3] /= lengths_mult
    #score[3:] /= angles_mult
    z = np.random.normal(size=lattice.shape)
    noise = step**0.5 * z
    penalty = np.zeros_like(lattice)
    #penalty[3:] = -l * step**0.5 * lattice[3:] * np.abs(lattice[3:])
    print(f"Lattice: Sigma: {sigma}, Lattice: {lattice}, Noise_mult: {step**0.5}, Score_mult: {0.5*step/sigma**0.5}")
    lattice = lattice - 0.5*step*score + .2*noise + penalty
    lattice = expand(lattice, lattice_system) #mask@lattice
    #lattice[:3] = torch.maximum(lattice[:3], torch.tensor(2))
    lattice9 = B6_to_B9(B6_inv_transform(lattice), 'cpu', np.linalg.det(atoms.lattice_mat)).cpu().detach().numpy()
    randomized_atoms = Atoms(coords = atoms.frac_coords,
                             lattice_mat = lattice9,
                             elements = atoms.elements,
                             cartesian = False)
    print(f"det: {np.linalg.det(atoms.lattice_mat)}")
    return randomized_atoms, lattice9, randomized_atoms.cart_coords

def positions_step(xt, atoms, operations, space_group, sg_type, sigma, step, t, cp_matrix, model, device):
    z = np.random.normal(size=xt.shape)
    element_prob = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='atoms')[2].detach().cpu().numpy()
    print([atomic_numbers_to_symbols(line[-6:]+1) for line in np.argsort(element_prob, axis=1)])
    score = get_output(atoms, operations, space_group, sg_type, model, t, device, output_type='edges')[0] / sigma**0.5        
    score = score.cpu().detach().numpy()
    print(f"Pos: Sigma: {sigma}, Noise_mult: {step**0.5}, Score_mult: {0.5*step/sigma**0.5}")
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
                      gradient,
                      T=60,
                      epsilon=1e-3,
                      lattice_epsilon=1e-2,
                      noise_original_positions=False,
                      random_positions=True,
                      random_lattice=True):
    #lattice = (3 + 2.* np.abs(np.random.normal(size=(3,)))) * np.eye(3)
    #lattice = atoms.lattice_mat
    if random_lattice:
        lattice = B6_to_B9(B6_inv_transform(sample_lattice(space_group, atoms, len(operations)))).numpy()
    else:
        lattice = atoms.lattice_mat
    lattice_system = get_lattice_system(space_group)
    cp_matrix = c_to_p_matrix(sg_type, lattice_system).numpy()
    mask_indices = get_mask(lattice_system)
    #mask = torch.tensor(get_noise_mask(lattice_system), dtype=lattice.dtype)
    mask = get_noise_mask(lattice_system).numpy()
    primitive_lattice = cp_matrix@lattice
    atoms = Atoms(coords = atoms.frac_coords@primitive_lattice,
                  lattice_mat = lattice,
                  elements = atoms.elements,
                  cartesian = True)
    if random_positions:
        xt = get_sampling_coords(atoms, noise_original_positions, 0.15)
    else:
        xt = atoms.cart_coords
    result = [atoms]
    
    output_atoms = Atoms(coords = xt,
                             lattice_mat = cp_matrix@lattice,
                             elements = atoms.elements,
                             cartesian = True)
    apply_operations_atoms(output_atoms, operations, 4e-1)[0].write_cif(f"0_original.cif", with_spg_info=False)
    
    for t in range(len(b)-1, -1, -1):
        sigma = b[t]
        step = epsilon * (sigma / b[0])
        step = step.cpu().detach().numpy()
        lattice_sigma = lattice_b[t]
        l_step = lattice_epsilon * (lattice_sigma / lattice_b[0])
        l_step = l_step.cpu().detach().numpy()
        
        #print(f"Time: {t}, Step size: {(step, l_step)}")
        
        for i in range(T):        
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
                randomized_atoms, lattice, xt = lattice_step(randomized_atoms, operations, space_group, sg_type, mask_indices, mask, lattice_sigma, l_step, t_tensor, model, device, gradient=gradient)
        output_atoms = Atoms(coords = xt,
                             lattice_mat = cp_matrix@lattice,
                             elements = atoms.elements,
                             cartesian = True)
        result.append(apply_operations_atoms(output_atoms, operations, 4e-1)[0])
    return result, None
   
