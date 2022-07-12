import torch
from functions.atomic import return_to_lattice, get_output, B6_transform, B6_inv_transform, get_sampling_coords, B9_to_B6, B6_to_B9
from jarvis.core.atoms import Atoms
import numpy as np
from collections import Counter
from datasets.symmetries import get_operations, reduce_atoms, apply_operations_atoms

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
    
def lattice_step(atoms, operations, sigma, step, t, model, device, gradient, lengths_mult, angles_mult, l=0.1):
    lattice = B6_transform(B9_to_B6(torch.tensor(atoms.lattice_mat)))
    score = None
    while score == None:
        score = get_output(atoms, operations, model, t, device, output_type='lattice', b6=lattice, gradient=gradient, emax=200)[1].cpu()
    score /= sigma.item()**0.5
    score[:3] /= lengths_mult
    score[3:] /= angles_mult
    z = np.random.normal(size=(6,))
    noise = step**0.5 * z
    penalty = np.zeros(6)
    #penalty[3:] = -l * step**0.5 * lattice[3:] * np.abs(lattice[3:])
    print(sigma, lattice, noise, -0.5*step*score, penalty)
    lattice = lattice - 0.5*step*score + noise + penalty
    #lattice[:3] = torch.maximum(lattice[:3], torch.tensor(2))
    lattice9 = B6_to_B9(B6_inv_transform(lattice), 'cpu', np.linalg.det(atoms.lattice_mat)).cpu().detach().numpy()
    randomized_atoms = Atoms(coords = atoms.frac_coords,
                             lattice_mat = lattice9,
                             elements = atoms.elements,
                             cartesian = False)
    print("det", np.linalg.det(atoms.lattice_mat))
    return randomized_atoms, lattice9, randomized_atoms.cart_coords

def positions_step(xt, atoms, operations, sigma, step, t, model, device):
    z = np.random.normal(size=xt.shape)
    score = get_output(atoms, operations, model, t, device, output_type='edges')[0] / sigma**0.5        
    score = score.cpu().detach().numpy()
    xt = xt - 0.5*step*score + step**0.5 * z
    xt = return_to_lattice(xt, atoms.lattice_mat)
    atoms = Atoms(coords = xt,
                  lattice_mat = atoms.lattice_mat,
                  elements = atoms.elements,
                  cartesian = True)
    return atoms, xt

@torch.no_grad()
def langevin_dynamics(atoms, phonopy_file, b, model, device, gradient, lengths_mult, angles_mult, T=40, epsilon=1e-4, noise_original_positions=False):
    b_iter = reversed(list(enumerate(b)))
    #lattice = (3 + 2.* np.abs(np.random.normal(size=(3,)))) * np.eye(3)
    lattice = atoms.lattice_mat
    atoms = Atoms(coords = atoms.frac_coords,
                  lattice_mat = lattice,
                  elements = atoms.elements,
                  cartesian = False)
    operations = get_operations(phonopy_file)
    atoms, _ = reduce_atoms(atoms, operations)
    xt = get_sampling_coords(atoms, noise_original_positions, 0.15)
    #xt = atoms.frac_coords @ lattice
    result = [atoms]
    for t, sigma in b_iter:
        step = epsilon * (sigma / b[0])
        step = step.cpu().detach().numpy()
        print("Time: {}, Step size: {}".format(t, step))
        randomized_atoms = Atoms(coords = xt,
                                 lattice_mat = lattice,
                                 elements = atoms.elements,
                                 cartesian = True)
        result.append(randomized_atoms)
        
        for i in range(T):
            t_tensor = torch.ones(1).to(device)*t
            randomized_atoms, xt = positions_step(xt, randomized_atoms, operations, sigma, step, t_tensor, model, device)
            #if 10 > t:
            #    randomized_atoms, lattice, xt = lattice_step(randomized_atoms, operations, sigma, step, t_tensor, model, device, gradient=gradient, lengths_mult=lengths_mult, angles_mult=angles_mult)
    result.append(apply_operations_atoms(randomized_atoms, operations, 2e-1)[0])
    return result, None
   
