import torch
import numpy as np
import dgl.function as fn
import dgl
import sympy
from functions.graph import atom_dgl_multigraph
from jarvis.core.atoms import Atoms
from random import sample, shuffle
from joblib import Parallel, delayed
from functions.lattice import degrees_of_freedom, get_noise_mask, get_lattice_system, c_to_p, c_to_p_matrix, expand, reduce
import logging
from time import time


def cotan(x):
    return torch.tan(0.5*np.pi - x)

def acotan(x):
    return 0.5*np.pi - torch.atan(x)

def softplus(x, k=10., t=100.):
    result = torch.clone(x)
    result[result < t] = k*torch.log(1 + torch.exp(result[result < t]/k))
    return result
 
def softplus_inv(x, k=10., t=100.):
    result = torch.clone(x)
    result[result < t] = k*torch.log(torch.exp(result[result < t]/k) - 1)
    return result

def B6_transform(B6):
    # (a, b, c, alpha, beta, gamma) -> basis R^6 representation
    if type(B6) is not torch.Tensor:
        B6 = torch.tensor(B6)
    assert B6.shape == (6,)
    result = B6.clone()
    result[:3] = softplus_inv(B6[:3])
    max_gamma = torch.minimum(B6[3] + B6[4], 2*np.pi - B6[3] - B6[4])
    B6_6 = cotan(np.pi * (B6[5] - torch.abs(B6[3] - B6[4])) / (max_gamma - torch.abs(B6[3] - B6[4])))
    result[3:5] = cotan(B6[3:5])
    result[5] = B6_6
    return result

def B6_inv_transform(B6):
    if type(B6) is not torch.Tensor:
        B6 = torch.tensor(B6)
    assert B6.shape == (6,)
    result = B6.clone()
    result[:3] = softplus(B6[:3])
    result[3:5] = acotan(B6[3:5])
    alpha, beta = acotan(B6[3]), acotan(B6[4])
    max_gamma = torch.minimum(alpha + beta, 2*np.pi - alpha - beta)
    result[5] = acotan(B6[5]) / np.pi * (max_gamma - torch.abs(alpha-beta)) + torch.abs(alpha-beta)
    return result
    
def B9_to_B6(B9, device='cpu'):
    # 3x3 matrix -> (a, b, c, alpha, beta, gamma)
    if type(B9) is not torch.Tensor:
        B6 = torch.tensor(B9)
    assert B9.shape == (3,3)
    lengths = torch.norm(B9, dim=1)
    result = torch.empty((6,), dtype=B9.dtype, device=device)
    result[[0,1,2]] = lengths
    result[3] = torch.acos(torch.dot(B9[2], B9[1]) / lengths[2] / lengths[1])
    result[4] = torch.acos(torch.dot(B9[0], B9[2]) / lengths[0] / lengths[2])
    result[5] = torch.acos(torch.dot(B9[1], B9[0]) / lengths[1] / lengths[0])
    return result    

def B6_to_B9(B6, device='cpu', so=1.):
    # (a, b, c, alpha, beta, gamma) -> 3x3 matrix
    # if impossible, return None
    # orientation determined by so parameter
    if type(B6) is not torch.Tensor:
        B6 = torch.tensor(B6)
    assert B6.shape == (6,)
    if torch.any(B6[:3] <= 0):
        return None
    result = torch.zeros((3,3), dtype=B6.dtype, device=device)
    B6 = B6.flatten()
    result[0, 0] = B6[0]
    result[1, 0] = B6[1] * torch.cos(B6[5])
    result[1, 1] = B6[1] * torch.sin(B6[5])
    b_20 = B6[2] * torch.cos(B6[4])
    b_21 = B6[2] * (torch.cos(B6[3]) - torch.cos(B6[5]) * torch.cos(B6[4])) / torch.sin(B6[5])
    result[2, 0] = b_20
    result[2, 1] = b_21
    x = B6[2].square() - b_20.square() - b_21.square()
    if x > 0:
        result[2, 2] = torch.sqrt(x)
        if torch.linalg.det(result) * so < 0:
            result[2, 2] = -torch.sqrt(x)
        return result
    return None

def equal(a, b, eps=1e-3):
    return np.abs(a-b) < eps

def get_permutation(lattice_system, lattice):
    b6 = B9_to_B6(lattice)
    if lattice_system == 'triclinic':
        return (0, 1, 2)
    elif lattice_system == 'monoclinic':
        if equal(b6[3], b6[4]):
            return (0, 1, 2)
        elif equal(b6[3], b6[5]):
            return (0, 2, 1)
        elif equal(b6[4], b6[5]):
            return (1, 2, 0)
        else:
            raise ValueError("Invalid lattice")
    elif lattice_system == 'orthorhombic':
        assert (equal(b6[3], np.pi/2)
                    and equal(b6[4], np.pi/2)
                    and equal(b6[5], np.pi/2))
        return (0, 1, 2)
    elif lattice_system == 'tetragonal':
        assert (equal(b6[3], np.pi/2)
                    and equal(b6[4], np.pi/2)
                    and equal(b6[5], np.pi/2))
        if equal(b6[0], b6[1]):
            return (0, 1, 2)
        elif equal(b6[0], b6[2]):
            return (0, 2, 1)
        elif equal(b6[1], b6[2]):
            return (1, 2, 0)
        else:
            raise ValueError("Invalid lattice")
    elif lattice_system == 'hexagonal':
        if equal(b6[0], b6[1]):
            assert (equal(b6[3], np.pi/2)
                    and equal(b6[4], np.pi/2)
                    and equal(b6[5], 2*np.pi/3))
            return (0, 1, 2)
        elif equal(b6[0], b6[2]):
            assert (equal(b6[3], np.pi/2)
                    and equal(b6[5], np.pi/2)
                    and equal(b6[4], 2*np.pi/3))
            return (0, 2, 1)
        elif equal(b6[1], b6[2]):
            assert (equal(b6[4], np.pi/2)
                    and equal(b6[5], np.pi/2)
                    and equal(b6[3], 2*np.pi/3))
            return (1, 2, 0)
        else:
            raise ValueError("Invalid lattice")
    elif lattice_system == 'cubic':
        assert (equal(b6[0], b6[1])
                    and equal(b6[0], b6[2])
                    and equal(b6[3], np.pi/2)
                    and equal(b6[4], np.pi/2)
                    and equal(b6[5], np.pi/2))
        return (0, 1, 2)
    else:
        raise ValueError(f"Invalid lattice system: {lattice_system}")

def get_sampling_coords(atoms, noise_original, noise_std):
    lattice = atoms.lattice_mat
    if noise_original:
        x = atoms.cart_coords
        x += noise_std * np.random.normal(size=x.shape)
        x = return_to_lattice(x, lattice)
    else:
        x = np.random.uniform(size=atoms.coords.shape)
        x = x @ lattice
    return x

def _get_eps(edge_movement, graph):
    graph = dgl.reverse(graph.local_var(), copy_ndata=True, copy_edata=True)
    norm = torch.linalg.norm(graph.edata['r'], dim=1).view(-1,1).repeat(1, 3)
    graph.edata['edge_movement'] = 15 * graph.edata['r'] / norm * torch.tanh(edge_movement / 15)
    assert graph.edata['edge_movement'].shape == graph.edata['r'].shape
    graph.update_all(fn.copy_e('edge_movement', 'message'), fn.sum('message', 'movement'))
    return graph.ndata['movement']

def lengths(b9, v):
    v = v@b9
    return torch.sqrt((v*v).sum(axis=1))
    
def empiric_gradient(b6, v, space_group, device='cpu', delta_val=0.1):
    lattice_system = get_lattice_system(space_group)
    result = torch.empty((v.shape[0], b6.shape[0]), device=device)
    #b9 = B6_to_B9(B6_inv_transform(expand(b6, lattice_system)), device=device).to(dtype=v.dtype)
    #l = lengths(b9, v)
    for i in range(b6.shape[0]):
        delta = torch.zeros_like(b6)
        delta[i] = delta_val
        b9_delta_plus = B6_to_B9(B6_inv_transform(expand(b6 + delta, lattice_system)), device=device).to(dtype=v.dtype)
        b9_delta_minus = B6_to_B9(B6_inv_transform(expand(b6 - delta, lattice_system)), device=device).to(dtype=v.dtype)
        result[:, i] = (lengths(b9_delta_plus, v) - lengths(b9_delta_minus, v)) / (2*delta_val)
    return result

@torch.no_grad()
def get_gradients(r, space_group, b6, lattice, device, emax, dtype):
    inv_lattice = np.linalg.inv(lattice)
    r_numpy = r.cpu().detach().numpy()
    b6_numpy = b6.cpu().detach().numpy()
    #index = np.arange(graph.edata['equiv'].shape[0])
    #index = (graph.edata['r'].squeeze(-1).detach().cpu().numpy() > 0.15).ravel()
    index = np.arange(r.shape[0])
    if len(index) == 0:
        return None, index
    if emax > 0 and len(index) > emax:
        index[sample(list(np.argwhere(index)[:,0]), len(index) - emax)] = False
    r_numpy = r_numpy[index]
    inv_lattice_t = torch.linalg.inv(torch.tensor(lattice, device=device, dtype=r.dtype))
    lattice_system = get_lattice_system(space_group)
    gradients = empiric_gradient(reduce(b6, lattice_system), r[index]@inv_lattice_t, space_group, device=device)
    gradients = gradients / (1e-3 + gradients.square().sum(dim=1, keepdim=True))
    return gradients, index

def get_graphs(atoms, operations, device, t, lattice_system, sg_type):
    atoms = c_to_p(atoms, sg_type, lattice_system)
    #atoms, equiv = apply_operations_atoms(atoms, operations)
    #graph, line_graph = MyGraph.atom_dgl_multigraph(atoms, min_neighbours=16, random_neighbours=6)
    graph, line_graph = atom_dgl_multigraph(atoms=atoms, operations=operations)
    graph, line_graph = graph.to(device=device), line_graph.to(device=device)
    graph.ndata['step'] = t.to(dtype=int) * torch.ones([graph.number_of_nodes(),], device=device, dtype=torch.int)
    #graph.edata['equiv'] = get_equiv(equiv, graph, device)
    return graph, line_graph

def get_output(noised_atoms, operations, space_group, sg_type, model, t, device, output_type='all', emax=200):
    graphs, line_graphs, b6, b9, lattice_system, noise_mask = [], [], [], [], [], []
    for i in range(len(noised_atoms)):
        b6.append(B6_transform(B9_to_B6(torch.tensor(noised_atoms[i].lattice_mat, device=device), device=device)))
        b9.append(noised_atoms[i].lattice_mat)
        lattice_system.append(get_lattice_system(space_group[i]))
        noise_mask.append(get_noise_mask(lattice_system[i], device=device))
        graph, line_graph = get_graphs(noised_atoms[i], operations[i], device, t[i], lattice_system[i], sg_type[i])
        graphs.append(graph)
        line_graphs.append(line_graph)
    batch_graph = dgl.batch(graphs)
    batch_line_graph = dgl.batch(line_graphs)
    output = model(batch_graph, batch_line_graph)
    eps_atoms, eps_lattice = [], []
    start_i = 0
    for i in range(len(noised_atoms)):
        end_i = start_i + graphs[i].edges()[0].shape[0]
        if output_type == 'all' or output_type == 'edges':
            eps_atoms.append(_get_eps(output[start_i:end_i,0].unsqueeze(-1), graphs[i]))
        if output_type == 'all' or output_type == 'lattice':
            gradient_mat, index = get_gradients(graphs[i].edata['r'],
                                                space_group[i],
                                                b6[i],
                                                b9[i],
                                                device=device,
                                                emax=emax,
                                                dtype=output.dtype)
            if gradient_mat == None or torch.isnan(output[:,1]).any():
                return None, None
            gradient_mat = gradient_mat.to(dtype=noise_mask[i].dtype)
            eps_lattice.append((output[index+start_i,1].unsqueeze(-1) * gradient_mat).mean(axis=0))
        start_i = end_i
    return eps_atoms, eps_lattice
    
def return_to_lattice(x, lattice, space_group, sg_type):
    lattice_system = get_lattice_system(space_group)
    lattice = c_to_p_matrix(sg_type, lattice_system=='monoclinic').detach().numpy()@lattice
    relative = x @ np.linalg.inv(lattice)
    relative = relative % 1.
    return relative @ lattice
