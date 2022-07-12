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
#import time


def cotan(x):
    return torch.tan(0.5*np.pi - x)

def acotan(x):
    return 0.5*np.pi - torch.atan(x)
    
def softplus(x, t=10.):
    result = torch.clone(x)
    result[result < t] = torch.log(1 + torch.exp(result[result < t]))
    return result
 
def softplus_inv(x, t=10.):
    result = torch.clone(x)
    result[result < t] = torch.log(torch.exp(result[result < t]) - 1)
    return result

def B6_transform(B6):
    # (a, b, c, alpha, beta, gamma) -> basis R^6 representation
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
    
def worker(job_b, job_v, gradient):
    result = np.empty((job_v.shape[0], 6))
    for i in range(job_v.shape[0]):
        result[i] = gradient(job_b, job_v[i]).squeeze(1)
    return result

def compute_gradients(b6, v, gradient, n_jobs=10):
    n = v.shape[0]
    p = n//n_jobs
    ids = list(range(n))
    bounds = [ids[p*i:min(p*(i+1), n)] for i in range(int(np.ceil(n/p)))]
    result = np.vstack(Parallel(n_jobs=n_jobs)(delayed(worker)(b6, v[bound], gradient) for bound in bounds))
    return result    

def mean_gradient_estimate(graph, b6, edge_movement, lattice, gradient, device, emax):
    inv_lattice = np.linalg.inv(lattice)
    r_numpy = graph.edata['r'].cpu().detach().numpy()
    b6_numpy = b6.cpu().detach().numpy()
    if emax > 0 and r_numpy.shape[0] > emax:
        index = sample(list(range(r_numpy.shape[0])), emax)
        r_numpy = r_numpy[index]
        edge_movement = edge_movement[index]
    #print(np.abs(edge_movement.cpu().detach().numpy()).max())
    stability = 0.001
    gradients = compute_gradients(b6_numpy, stability * r_numpy @ inv_lattice, gradient)
    #gradients = np.array([gradient(b6_numpy, stability * r_numpy[i] @ inv_lattice) for i in range(r_numpy.shape[0])]).squeeze(2)
    #gradients = [get_b6_grad(b6_numpy, stability * r_numpy[i] @ inv_lattice, device) for i in range(r_numpy.shape[0])]#.squeeze(2)
    #print(gradient(b6_numpy, stability * r_numpy[0] @ inv_lattice), get_b6_grad(b6_numpy, stability * r_numpy[0] @ inv_lattice))
    #data_start = time.time()
    #np.array([gradient(b6_numpy, stability * r_numpy[i] @ inv_lattice) for i in range(r_numpy.shape[0])]).squeeze(2)
    #print("sympy: ", time.time()-data_start)
    #data_start = time.time()
    #[get_b6_grad(b6_numpy, stability * r_numpy[i] @ inv_lattice, device) for i in range(r_numpy.shape[0])]
    #print("torch: ", time.time()-data_start)
    #gradients = np.empty((r_numpy.shape[0], 6))
    #for i in range(r_numpy.shape[0]):
    #    gradients[i] = gradient(b6_numpy, r_numpy[i] @ inv_lattice).squeeze(1)
        #if np.isnan(gradients[i]).any() or not -10e10 < np.linalg.norm(gradients[i]) < 10e10:
        #    print("b6: ", b6)
        #    print("v: ", r_numpy[i] @ inv_lattice)
    #gradients = torch.stack(gradients)
    gradients = torch.tensor(gradients, device=device)
    gradients /= stability
    #if torch.isnan(gradients).any:
    #    print("b6: ", b6)
    result = edge_movement * gradients / gradients.square().sum(dim=1, keepdim=True)
    #print(result.mean(dim=0))
    #print("b6: ", b6)
    return result.mean(dim=0)
    

def get_graphs(atoms, device, t):
    graph, line_graph = MyGraph.atom_dgl_multigraph(atoms, min_neighbours=12, random_neighbours=8)
    graph, line_graph = graph.to(device=device), line_graph.to(device=device)
    graph.ndata['step'] = t * torch.ones([graph.number_of_nodes(),], device=device, dtype=torch.int)
    return graph, line_graph

def get_output(noised_atoms, model, t, device, b6=None, gradient=None, output_type='all', emax=50):
    graph, line_graph = get_graphs(noised_atoms, device, t)
    lattice = torch.from_numpy(noised_atoms.lattice_mat).to(device).float()
    output = model(graph, line_graph)
    #assert fake_probabilities.shape == (graph.nodes().shape[0], 1)
    #assert eps_edge.shape == (graph.edges()[0].shape[0], 1)
    eps_atoms, eps_lattice = None, None
    if output_type == 'all' or output_type == 'edges':
        eps_atoms = _get_eps(output[:,0].unsqueeze(-1), graph)
    if output_type == 'all' or output_type == 'lattice':
        eps_lattice = mean_gradient_estimate(graph,
                                             b6,
                                             output[:,1].unsqueeze(-1),
                                             noised_atoms.lattice_mat,
                                             gradient,
                                             device=device,
                                             emax=emax)
    return eps_atoms, eps_lattice
    
def return_to_lattice(x, lattice):
    relative = x @ np.linalg.inv(lattice)
    relative = relative % 1.
    return relative @ lattice
