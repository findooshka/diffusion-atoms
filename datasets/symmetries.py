import os
import torch
import numpy as np
from jarvis.core.atoms import Atoms

def get_vector(output, i):
    start = output[i].find('[')
    end = output[i].find(']')
    line = output[i][start+1:end].split(',')
    line = np.array([float(number.strip()) for number in line])
    return line

def get_matrix(output, i):
    return np.array([get_vector(output, i), get_vector(output, i+1), get_vector(output, i+2)])
    
def get_operations(filename):
    with open(filename, 'r') as f:
        output = f.read().split("\n")
    assert output[4] == 'space_group_operations:'
    i = 5
    operations = []
    while output[i].find('rotation') >= 0:
        operations.append((get_matrix(output, i+1), get_vector(output, i+4)))
        i += 5
    sg_type = output[1].split()[-1][1]
    space_group = int(output[2].split()[-1])
    #return operations, 1, 'P'
    return operations, space_group, sg_type
    #return [(np.eye(3), np.zeros(3))], space_group, sg_type
    #return [(np.eye(3), np.zeros(3))], 1, 'P'
    
def apply_operation(operation, frac_coords):
    return ((operation[0]@frac_coords.T).T + operation[1]) % 1

def apply_operations(operations, frac_coords):
    return np.vstack([apply_operation(operation, frac_coords) for operation in operations])

def crystal_distance(pos1, pos2, norm=True):
    pos1 = pos1.reshape(-1, 1, 3)
    pos2 = pos2.reshape(1, -1, 3)
    diff = (pos1 - pos2)%1
    diff = np.stack((diff, 1-diff))
    diff = diff.min(axis=0)
    if norm:
        return np.linalg.norm(diff, axis=-1)
    return diff

def reduce_atoms(atoms, operations, check_consistency=False):
    indices = []
    for i in range(len(atoms.elements)):
        equiv_pos = apply_operations(operations, atoms.frac_coords[i])
        if (crystal_distance(equiv_pos, atoms.frac_coords[indices]) > 1e-4).all():
            indices.append(i)
        #print(operations[0])
        #print(equiv_pos)
        #print(crystal_distance(equiv_pos, atoms.frac_coords))
        if check_consistency and (crystal_distance(equiv_pos, atoms.frac_coords).min(axis=1) > 1e-4).any():
            raise ValueError("Inconsistent strcture/operations")
    return (
                Atoms(coords=atoms.frac_coords[indices],
                      lattice_mat=atoms.lattice_mat,
                      elements=np.array(atoms.elements)[indices],
                      cartesian=False),
                indices,
           )

def apply_operations_atoms(atoms, operations, repeat_threshold=-1):
    n = len(atoms.elements)
    coords = apply_operations(operations, atoms.frac_coords)
    elements = list(np.array(atoms.elements)) * len(operations)
    if repeat_threshold > 0:
        index = [0]
        for i in range(1, len(elements)):
            if (crystal_distance(coords[i], coords[i%n:i:n]) > repeat_threshold).all():
                index.append(i)
        coords = coords[index]
        elements = list(np.array(elements)[index])
    return (
                Atoms(coords=coords,
                      lattice_mat=atoms.lattice_mat,
                      elements=elements,
                      cartesian=False),
                list(range(n)) * len(operations)
           )

def get_equiv(indices, g, device):
    torch_indices = torch.tensor(indices, device=device)
    equiv = torch.zeros_like(g.edges()[0])
    edge_indices = torch_indices[g.edges()[0]] == torch_indices[g.edges()[1]]
    equiv[edge_indices] = 1
    return equiv.unsqueeze(1)
    
