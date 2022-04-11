import torch
import numpy as np
import dgl.function as fn
from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
from random import sample, shuffle


def cotan(x):
    return torch.tan(0.5*np.pi - x)

def acotan(x):
    return 0.5*np.pi - torch.atan(x)
    
def softplus(x, t=15.):
    result = torch.clone(x)
    result[result < t] = torch.log(1 + torch.exp(result[result < t]))
    return result
 
def softplus_inv(x, t=15.):
    result = torch.clone(x)
    result[result < t] = torch.log(torch.exp(result[result < t]) - 1)
    return result

def B6_transform(B6):
    assert B6.shape == (6,)
    B6[:3] = softplus_inv(B6[:3])
    B6[3:] = cotan(B6[3:])
    return B6
    #return torch.tensor(softplus(B6[0]), softplus(B6[1]), softplus(B6[2]),
    #                    cotan(B6[3]), cotan(B6[4]), cotan(B6[5]), device=B6)

def B6_inv_transform(B6):
    assert B6.shape == (6,)
    B6[:3] = softplus(B6[:3])
    B6[3:] = acotan(B6[3:])
    return B6

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
    

def get_graphs(atoms, device):
    graph, line_graph = Graph.atom_dgl_multigraph(atoms, max_neighbors=20)
    graph, line_graph = graph.to(device=device), line_graph.to(device=device)
    return graph, line_graph

def get_output(noised_atoms, model, t, device, s_t=None, output_type='all'):
    graph, line_graph = get_graphs(noised_atoms, device)
    lattice = torch.from_numpy(noised_atoms.lattice_mat).to(device).float()
    eps_edge, fake_probabilities, eps_lattice = model(graph, line_graph, lattice, t, output_type=output_type)
    #assert fake_probabilities.shape == (graph.nodes().shape[0], 1)
    #assert eps_edge.shape == (graph.edges()[0].shape[0], 1)
    eps_atoms = None
    if output_type == 'all' or output_type == 'edges':
        eps_atoms = _get_eps(eps_edge, graph)
    if output_type == 'all' or output_type == 'lattice':
        eps_lattice = _get_eps(eps_lattice, line_graph, True)
        line_graph.edata['r'] = line_graph.ndata['r'].index_select(0, line_graph.edges()[1])
        eps_lattice = get_L_inv_estimate(graph, line_graph, eps_lattice, device=device, s_t=s_t)
    return eps_atoms, fake_probabilities, eps_lattice
    
def return_to_lattice(x, lattice):
    relative = x @ np.linalg.inv(lattice)
    relative = relative % 1.
    return relative @ lattice
