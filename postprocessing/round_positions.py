import logging
from argparse import ArgumentParser
import os
import torch
import numpy as np
#import scipy as sp
from scipy.linalg import lstsq, null_space, eig
from jarvis.core.atoms import Atoms
from tqdm import tqdm

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
    
def inverse_operation(operation):
    M_inverse = np.linalg.inv(operation[0])
    return (M_inverse, -M_inverse@operation[1])
    
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

def reduce_atoms(atoms, operations, check_consistency=False, tol=1e-4):
    indices = []
    for i in range(len(atoms.elements)):
        equiv_pos = apply_operations(operations, atoms.frac_coords[i])
        if (crystal_distance(equiv_pos, atoms.frac_coords[indices]) > tol).all():
            indices.append(i)
        #print(operations[0])
        #print(equiv_pos)
        #print(crystal_distance(equiv_pos, atoms.frac_coords))
        if check_consistency and (crystal_distance(equiv_pos, atoms.frac_coords).min(axis=1) > tol).any():
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
            if (np.linalg.norm(crystal_distance(coords[i], coords[i%n:i:n], norm=False) @ atoms.lattice_mat, axis=-1) > repeat_threshold).all():
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

def test(atoms, operations):
    coords = apply_operations(operations, atoms.frac_coords)
    elements = list(np.array(atoms.elements)) * len(operations)
    for i in range(len(elements)):
        for j in range(i):
            if crystal_distance(coords[i], coords[j]) < 4e-2 and elements[i] != elements[j]:
                return True
    return False

def get_equiv(indices, g, device):
    torch_indices = torch.tensor(indices, device=device)
    equiv = torch.zeros_like(g.edges()[0])
    edge_indices = torch_indices[g.edges()[0]] == torch_indices[g.edges()[1]]
    equiv[edge_indices] = 1
    return equiv.unsqueeze(1)

def get_subspace(operation,
                 delta=np.array([0, 0, 0], dtype=float)):
    solution_M = operation[0] - np.eye(3)
    solution_y = operation[1] + delta
    solution, _, _, _  = lstsq(solution_M, solution_y)
    if (np.abs(solution_M @ solution - solution_y) > 1e-4).any():
        return None, None
    return (null_space(operation[0] - np.eye(3)),
            solution)

def projection(M):
    #print(M.T @ M)
    return M @ np.linalg.inv(M.T @ M) @ M.T

def project(subspace, x):
    return projection(subspace[0]) @ (x-subspace[1]) + subspace[1]

def distance_to_subspace(subspace, x):
    return np.max(np.abs(project(subspace, x) - x))

def get_subspace_close_by(operation, x, threshold, search_width=3):
    dist = (operation[0] @ x + operation[1] - x) % 1
    min_dist = (np.abs(np.stack((dist, 1-dist), axis=-1))).min(axis=-1)
    if (min_dist > threshold).any():
        return None, None
    for delta_x in range(-search_width, search_width+1):
        for delta_y in range(-search_width, search_width+1):
            for delta_z in range(-search_width, search_width+1):
                space = get_subspace(operation, np.array([delta_x, delta_y, delta_z], dtype=float))
                if space[0] is not None and distance_to_subspace(space, x) < threshold:
                    return space
    return None, None

def intersect(space1, space2, threshold=1e-4):
    if space1[0].shape[1] == 0 and distance_to_subspace(space2, space1[1]) < threshold:
        return space1
    if space2[0].shape[1] == 0 and distance_to_subspace(space1, space2[1]) < threshold:
        return space2
    A, c = space1
    B, d = space2
    projA = projection(A)
    projB = projection(B)
    projAB = projA @ projB
    vals, vectors = eig(projAB)
    null_space = vectors[:,np.where(np.abs(vals - 1) < 1e-4)[0]]  # kernel
    solution_M = np.eye(projA.shape[0]) - projAB
    solution_y = projAB @ (c-d) + projA @ (d-c)
    solution, _, _, _  = lstsq(solution_M, solution_y)
    if (np.abs(solution_M @ solution - solution_y) > 1e-4).any():
        return None, None
    return null_space, solution + c

def special_position(operations, x, threshold=4e-2):
    subspace = (np.eye(3), np.zeros(3))
    for operation in operations:
        new_subspace = get_subspace_close_by(operation, x, threshold=threshold)
        if new_subspace[0] is not None:
            intersection = intersect(subspace, new_subspace)
            if intersection[0] is not None and distance_to_subspace(intersection, x) < threshold:
                subspace = intersection
    return project(subspace, x)

def normal_positions(operations, frac_coords, threshold=2e-2):
    result = np.copy(frac_coords)
    for i, x in enumerate(frac_coords):
        symmetrized = False
        for j, y in enumerate(frac_coords[:i]):
            for operation in operations:
                x_sym = apply_operation(operation, x)
                inverse = inverse_operation(operation)
                dist = np.max(crystal_distance(x_sym, y, norm=False)[0,0])
                #min_i = np.argmin(dist)
                if dist < threshold:
                    result[i] = apply_operation(inverse, y)
                    symmetrized = True
                    break
            if symmetrized:
                break
    return result
            

def round_positions(operations, atoms):
    coords = np.array([special_position(operations, line) for line in atoms.frac_coords]).real
    coords = normal_positions(operations, coords)
    atoms = Atoms(coords=coords,
                  lattice_mat=atoms.lattice_mat,
                  elements=atoms.elements,
                  cartesian=False)
    return atoms

def save_to_dir(path, structs, vasp=False):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, struct in enumerate(structs):
        if vasp:
            Poscar(struct).write_file(os.path.join(path, str(i) + ".vasp"))
        else:
            CifWriter(struct).write_file(os.path.join(path, str(i) + ".cif"))
            
def process_batch(batch_dir, vasp, op_dir):
    batch_dir = batch_dir.rstrip(r'\/')
    output_dir = batch_dir + "_rounded"
    logging.info(f"Writting files to \"{output_dir}\"")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for sg_dir in tqdm(os.listdir(batch_dir)):
        operations, space_group, sg_type = get_operations(op_dir + sg_dir.split('_')[-1])
        if 'finals' in os.listdir(os.path.join(batch_dir, sg_dir)):
            read_dir_path = os.path.join(batch_dir, sg_dir, "finals")
        else:
            read_dir_path = os.path.join(batch_dir, sg_dir)
        structures = os.listdir(read_dir_path)
        out_sg_dir_path = os.path.join(output_dir, sg_dir)
        if not os.path.exists(out_sg_dir_path):
            os.makedirs(out_sg_dir_path)
        out_sg_dir_path = os.path.join(output_dir, sg_dir)
        if not os.path.exists(out_sg_dir_path):
            os.makedirs(out_sg_dir_path)
        for structure in structures:
            try:
                atoms = Atoms.from_cif(os.path.join(read_dir_path, structure), use_cif2cell=False, get_primitive_atoms=False)
                #atoms = reduce_atoms(atoms, operations)[0]
                #atoms = apply_operations_atoms(atoms, operations, 0.01)[0]
                atoms = round_positions(operations, atoms)
            except Exception as e:
                logging.error(f"Error while processing structure {os.path.join(read_dir_path, structure)}")
                logging.error(e)
            else:
                if vasp:
                    atoms.write_poscar(os.path.join(out_sg_dir_path, structure))
                else:
                    atoms.write_cif(os.path.join(out_sg_dir_path, structure), with_spg_info=False)

        
if __name__ == "__main__":
    logging.basicConfig(level=20)
    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True, help="Generation batch folder name")
    argparser.add_argument("--vasp", type=bool, default=False, help="Output in vasp format")
    argparser.add_argument("--sg_dir", type=str, default="/home/arsen/data/space_groups/", help="Space group operations folder")
    args = argparser.parse_args()
    process_batch(args.dir, vasp=args.vasp, op_dir=args.sg_dir)
