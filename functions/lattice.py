import torch
import numpy as np
import sympy as sb
from sympy.utilities.lambdify import lambdify
import os, subprocess
from jarvis.core.atoms import Atoms

def get_lattice_system(space_group):
    rhombohedral = set()#{146, 148, 155, 160, 161, 166, 167}
    if space_group in rhombohedral:
        return 'rhombohedral'
    elif 1 <= space_group <= 2:
        return 'triclinic'
    elif 3 <= space_group <= 15:
        return 'monoclinic'
    elif 16 <= space_group <= 74:
        return 'orthorhombic'
    elif 75 <= space_group <= 142:
        return 'tetragonal'
    elif 143 <= space_group <= 194:
        return 'hexagonal'
    elif 195 <= space_group <= 230:
        return 'cubic'
    else:
        raise ValueError(f"Invalid space group: {space_group}")        

def get_mask(lattice_system):
    if lattice_system == 'triclinic':
        return [0, 1, 2, 3, 4, 5]
    elif lattice_system == 'monoclinic':
        return [0, 1, 2, 5]
    elif lattice_system == 'orthorhombic':
        return [0, 1, 2]
    elif lattice_system == 'tetragonal':
        return [0, 2]
    elif lattice_system == 'hexagonal':
        return [0, 2]
    elif lattice_system == 'cubic':
        return [0]
    else:
        raise ValueError(f"Invalid lattice system: {lattice_system}")

def degrees_of_freedom(space_group):
    return len(get_mask(get_lattice_system(space_group)))
 
def get_noise_mask(lattice_system, device='cpu'):
    mask = get_mask(lattice_system)
    DoF = len(mask)
    result = torch.zeros((6, DoF), device=device)
    result[mask, range(DoF)] = 1
    if lattice_system == 'tetragonal' or lattice_system == 'hexagonal':
        result[1, 0] = 1
    elif lattice_system == 'cubic':
        result[1, 0] = 1
        result[2, 0] = 1
    return result
    
def c_to_p_matrix(sg_type, lattice_system, device='cpu'):
    if sg_type == 'C':
        result = torch.tensor([[0.5, -0.5, 0], [0.5, 0.5, 0], [0., 0., 1]], device=device)
    elif sg_type == 'A':
        result = torch.tensor([[0., 0., 1], [0.5, 0.5, 0], [0.5, -0.5, 0]], device=device)
    elif sg_type == 'I':
        result = torch.tensor([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]], device=device)
    elif sg_type == 'F':
        result = torch.tensor([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]], device=device)
    elif sg_type == 'R':
        result = 1/3 * torch.tensor([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]], device=device)
    elif sg_type == 'P':
        result = torch.eye(3, device=device)
    else:
        raise ValueError(f'Invalid group type: {sg_type}')
    if lattice_system == 'monoclinic':
        result = result[:,[0, 2, 1]]
    return result

def p_to_c_matrix(sg_type, lattice_system, device='cpu'):
    return torch.linalg.inv(c_to_p_matrix(sg_type, lattice_system, 'cpu'))

def c_to_p(atoms, sg_type, lattice_system):
    return Atoms(coords=atoms.cart_coords,
                 lattice_mat=c_to_p_matrix(sg_type, lattice_system).detach().numpy()@atoms.lattice_mat,
                 elements=atoms.elements,
                 cartesian=True)

def p_to_c(atoms, sg_type, lattice_system):
    # not adding equivalent sites
    return Atoms(coords=atoms.cart_coords,
                 lattice_mat=p_to_c_matrix(sg_type, lattice_system).detach().numpy()@atoms.lattice_mat,
                 elements=atoms.elements,
                 cartesian=True)


