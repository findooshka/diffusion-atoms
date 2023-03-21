import torch
import numpy as np
import dgl.function as fn
import dgl
import sympy
#from jarvis.core.graphs import Graph
from functions.graph import MyGraph
from jarvis.core.atoms import Atoms
from random import sample, shuffle
from joblib import Parallel, delayed
from datasets.symmetries import get_equiv, apply_operations_atoms
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
    #B6[3:] = cotan(B6[3:])
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
        if torch.linalg.det(result) * so < 0:
            result[2, 2] = -torch.sqrt(x)
        else:
            result[2, 2] = torch.sqrt(x)
        
        return result
    print('oh')
    #else:
    #    result = torch.zeros((3,3), device=device)
    #    penalty = penalty_mult * (1-x) / B6[2].square()
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

def get_b6_grad(b6, e, device='cpu'):
    if torch.is_tensor(b6):
        b6 = b6.detach()
    else:
        b6 = torch.tensor(b6, device=device)
    if torch.is_tensor(e):
        e = e.detach()
    else:
        e = torch.tensor(e, device=device)
    b6.requires_grad_()

    b6_raw = B6_inv_transform(b6)
    b9 = B6_to_B9(b6_raw, so=1, device=device)
    y = torch.matmul(e, b9)
    y = torch.norm(y)
    y.backward()

    return b6.grad

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

def add_random_atoms(init_atoms, l=0.2, scale_with_natoms=True):
    ##
    ## add atoms at random positions 
    ## elements are distributed as in the original structure
    ## added atom count ~ poisson(l)
    ##
    if scale_with_natoms:
        count = np.random.poisson(lam=l*len(init_atoms.elements))
    else:
        count = np.random.poisson(lam=l)
    new_coords = np.random.uniform(size=(count, 3))
    new_coords = new_coords @ init_atoms.lattice_mat
    new_coords = np.vstack((init_atoms.cart_coords, new_coords))
    new_elements = []
    for i in range(count):
        new_elements.append(sample(init_atoms.elements, 1)[0])
    new_elements = init_atoms.elements + new_elements
    fake = np.hstack((np.zeros(len(init_atoms.elements)), np.ones(count)))
    permutation = np.arange(len(new_elements), dtype=int)
    np.random.shuffle(permutation)
    fake = [fake[i] for i in permutation]
    new_elements = [new_elements[i] for i in permutation]
    new_coords = new_coords[permutation]
    return (
        Atoms(coords = new_coords,
              lattice_mat = init_atoms.lattice_mat,
              elements = new_elements,
              cartesian = True),
        fake
    )

def _get_eps(edge_movement, graph, line_graph=False):
    graph = graph.local_var()
    #graph.edata['edge_movement'] = graph.edata['r'] * edge_movement
    if line_graph:
        graph.edata['r'] = graph.ndata['r'].index_select(0, graph.edges()[1])
    norm = torch.linalg.norm(graph.edata['r'], dim=1).view(-1,1).repeat(1, 3)
    graph.edata['edge_movement'] = 15 * graph.edata['r'] / norm * torch.tanh(edge_movement / 15)
    assert graph.edata['edge_movement'].shape == graph.edata['r'].shape
    graph.update_all(fn.copy_edge('edge_movement', 'message'), fn.sum('message', 'movement'))
    #graph.update_all(fn.copy_edge('edge_movement', 'message'), fn.mean('message', 'movement'))
    return graph.ndata['movement']

def get_L_inv_estimate(graph, line_graph, edge_movement, device, s_t, p=3, max_d=8., det_threshold=0.1, max_attempts=30):
    # edge_movement: e*3 matrix of estimates of linear operator action on atom neighbours
    node_index = torch.randint(graph.nodes().shape[0], size=(1,), device=device)
    neighbour_indices = ((graph.edges()[1] == node_index) & (graph.edata['r'].norm(dim=1) < max_d)).nonzero()
    if neighbour_indices.numel() < p:
        print("Too few neighbours to estimate L")
        return None
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        chosen_edges = neighbour_indices[torch.randperm(neighbour_indices.numel())[:p]].flatten()
        X = line_graph.edata['r'].index_select(0, chosen_edges)
        X_unit = X / torch.linalg.norm(X, dim=1).view(p, 1)
        det = torch.linalg.det(X_unit)  #, torch.transpose(X_unit, 0, 1)))
        if det > det_threshold:
            break
    else:
        print("failed to find non-degenerate basis")
        return None
    lengths = torch.linalg.norm(X, dim=1).view(p, 1)
    movement = edge_movement.index_select(0, chosen_edges)
    movement = 0.1 * movement * lengths * s_t.sqrt()
    Y = X + movement
    Xt = torch.transpose(X, 0, 1)
    XXX = torch.linalg.inv(torch.matmul(Xt, X))
    XXX = torch.matmul(XXX, Xt)
    L = torch.matmul(XXX, Y)
    return L
    
def worker(job_b, job_v, space_group, gradient):
    result = np.empty((job_v.shape[0], degrees_of_freedom(space_group)))
    for i in range(job_v.shape[0]):
        result[i] = gradient(job_b, job_v[i], space_group).squeeze(1)
    return result

def compute_gradients(b6, v, space_group, gradient, n_jobs=10):
    n = v.shape[0]
    p = n//n_jobs
    ids = list(range(n))
    bounds = [ids[p*i:min(p*(i+1), n)] for i in range(int(np.ceil(n/p)))]
    result = np.vstack(Parallel(n_jobs=n_jobs)(delayed(worker)(b6, v[bound], space_group, gradient) for bound in bounds))
    #result = np.vstack((worker(b6, v[bound], space_group, gradient) for bound in bounds))
    return result    

def lengths(b9, v):
    v = v@b9
    return torch.sqrt((v*v).sum(axis=1))
    
def empiric_gradient(b6, v, space_group, device='cpu', delta_val=0.1):
    lattice_system = get_lattice_system(space_group)
    result = torch.empty((v.shape[0], b6.shape[0]), device=device)
    b9 = B6_to_B9(B6_inv_transform(expand(b6, lattice_system)), device=device).to(dtype=v.dtype)
    l = lengths(b9, v)
    for i in range(b6.shape[0]):
        delta = torch.zeros_like(b6)
        delta[i] = delta_val
        b9_delta_plus = B6_to_B9(B6_inv_transform(expand(b6 + delta, lattice_system)), device=device).to(dtype=v.dtype)
        b9_delta_minus = B6_to_B9(B6_inv_transform(expand(b6 - delta, lattice_system)), device=device).to(dtype=v.dtype)
        result[:, i] = (lengths(b9_delta_plus, v) - lengths(b9_delta_minus, v)) / (2*delta_val)
    return result

@torch.no_grad()
def get_gradients(graph, space_group, b6, lattice, gradient, device, emax, dtype):
    inv_lattice = np.linalg.inv(lattice)
    r_numpy = graph.edata['r'].cpu().detach().numpy()
    b6_numpy = b6.cpu().detach().numpy()
    index = list(np.argwhere((graph.edata['equiv'].squeeze(-1).detach().cpu().numpy() == 0)).ravel())
    if len(index) == 0:
        return None, index
    if emax > 0 and len(index) > emax:
        #index = sample(list(range(r_numpy.shape[0])), emax)
        index = sample(index, emax)
    r_numpy = r_numpy[index]
    #else:
    #    index = list(range(r_numpy.shape[0]))
    
    inv_lattice_t = torch.linalg.inv(torch.tensor(lattice, device=device, dtype=graph.edata['r'].dtype))
    lattice_system = get_lattice_system(space_group)
    gradients = empiric_gradient(reduce(b6, lattice_system), graph.edata['r'][index]@inv_lattice_t, space_group, device=device)
    
    #print(f"emp: {empiric_gradient(reduce(b6, lattice_system), graph.edata['r'][index]@inv_lattice_t, space_group, device=device)[-1]}")
    #print(b6)
    #print(graph.edata['r']@inv_lattice_t)
    #stability = 0.001
    #a_gradients = compute_gradients(b6_numpy, stability * r_numpy @ inv_lattice, space_group, gradient)
    #a_gradients = torch.tensor(a_gradients, device=device, dtype=dtype)
    #a_gradients /= stability
    #logging.info(f"grad_diff: {torch.abs(a_gradients - gradients).mean(axis=0)}")
    #logging.info(f"grad_abs: {torch.abs(gradients).mean(axis=0)}")
    #print((r_numpy @ inv_lattice)[-1])
    #print(f"analytic: {gradients[-1]}")
    return gradients / gradients.square().sum(dim=1, keepdim=True), index


from sklearn.linear_model import SGDRegressor
def gradient_estimate(gradients, target, device, alpha):
    #X = (gradients.T @ gradients + alpha * torch.eye(gradients.shape[1], device=device)).to(dtype=target.dtype)
    #return torch.inverse(X) @ gradients.T @ target
    estimator = SGDRegressor(loss='epsilon_insensitive', epsilon=0., fit_intercept=False, alpha=alpha, penalty='l2')
    estimator.fit(gradients.detach().cpu().numpy(), target.detach().cpu().numpy())
    return torch.tensor(estimator.coef_, dtype=target.dtype, device=device)

def gradient_estimate_gradient_estimate(gradients, target, device, alpha):
    differential = torch.empty()

def get_graphs(atoms, operations, device, t, lattice_system, sg_type):
    atoms = c_to_p(atoms, sg_type, lattice_system)
    atoms, equiv = apply_operations_atoms(atoms, operations)
    atoms.write_cif(with_spg_info=False)
    graph, line_graph = MyGraph.atom_dgl_multigraph(atoms, min_neighbours=16, random_neighbours=6)
    graph, line_graph = graph.to(device=device), line_graph.to(device=device)
    graph.ndata['step'] = t.to(dtype=int) * torch.ones([graph.number_of_nodes(),], device=device, dtype=torch.int)
    graph.edata['equiv'] = get_equiv(equiv, graph, device)
    return graph, line_graph

def get_output(noised_atoms, operations, space_group, sg_type, model, t, device, gradient=None, output_type='all', emax=200, L=None, s_t=None):
    b6 = B6_transform(B9_to_B6(torch.tensor(noised_atoms.lattice_mat, device=device), device=device))
    b9 = noised_atoms.lattice_mat
    lattice_system = get_lattice_system(space_group)
    noise_mask = get_noise_mask(lattice_system, device=device)
    graph, line_graph = get_graphs(noised_atoms, operations, device, t, lattice_system, sg_type)
    if output_type == 'atoms':
        graph.ndata['Z'] *= 0
    output, nodes_output = model(graph, line_graph)
    eps_atoms, eps_lattice, elements = None, None, None
    if output_type == 'all' or output_type == 'edges':
        eps_atoms = _get_eps(output[:,0].unsqueeze(-1), graph)
        eps_atoms = eps_atoms[:eps_atoms.shape[0]//len(operations)]
    if output_type == 'all' or output_type == 'lattice':
        #cart_r = graph.edata['r']
        #cart_diff = cart_r.norm(dim=1).to(dtype=b6.dtype) - (cart_r.to(dtype=b6.dtype)@L.to(dtype=b6.dtype)).norm(dim=1)
        #logging.info((cart_diff/cart_r.norm(dim=1).to(dtype=b6.dtype))[:10])
        #cart_diff = cart_diff.to(dtype=output.dtype) / s_t.sqrt()
        #cart_diff = cart_diff + torch.normal(torch.zeros_like(cart_diff))
        gradient_mat, index = get_gradients(graph,
                                            space_group,
                                            b6,
                                            b9,
                                            gradient,
                                            device=device,
                                            emax=emax,
                                            dtype=output.dtype)
        #logging.info(f"cart diff: {torch.abs(cart_diff).mean().item()}")
        #logging.info(f"{cart_diff.mean().item()}, {output[index,1].mean().item()}")
        #return (cart_diff[index] - output[index,1]).square().mean()
        if gradient_mat == None:
            return None, None
        gradient_mat = gradient_mat.to(dtype=noise_mask.dtype)
        eps_lattice = (output[index,1].unsqueeze(-1) * gradient_mat).mean(axis=0)
        #eps_lattice = (cart_diff[index].unsqueeze(-1) * gradient_mat).mean(axis=0)
        #eps_lattice = gradient_estimate(gradient_mat, cart_diff[index], device=device, alpha=.01)
        #eps_lattice = gradient_estimate(gradient_mat, output[index,1], device=device)
    if output_type == 'all' or output_type == 'atoms':
        elements = nodes_output[:nodes_output.shape[0]//len(operations)]
    return eps_atoms, eps_lattice, elements
    
def return_to_lattice(x, lattice, space_group, sg_type):
    lattice_system = get_lattice_system(space_group)
    lattice = c_to_p_matrix(sg_type, lattice_system=='monoclinic').detach().numpy()@lattice
    relative = x @ np.linalg.inv(lattice)
    relative = relative % 1.
    return relative @ lattice
