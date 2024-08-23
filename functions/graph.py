from jarvis.core.graphs import build_undirected_edgedata, compute_bond_cosines, canonize_edge, get_node_attributes, Graph
from collections import defaultdict
from typing import Optional
from jarvis.core.atoms import Atoms
from datasets.symmetries import apply_operations, inverse_operations_index
import numpy as np
import torch
import dgl
from random import sample
import logging
import itertools

def get_all_neighbors(atoms,
                      operations,
                      r=5,
                      bond_tol=0.15):
    """
    Get neighbors for each atom in the unit cell, out to a distance r.

    Contains [index_i, index_j, distance, image] array.
    """
    lattice_mat, frac_coords = atoms.lattice_mat, atoms.frac_coords
    n_op = len(operations)
    n_atoms = frac_coords.shape[0]
    frac_coords_sym = apply_operations(operations, frac_coords)
    inv = np.linalg.inv(lattice_mat).T
    recp_len = np.sqrt(np.sum(inv ** 2, axis=1))
    maxr = np.ceil((r + bond_tol) * recp_len)
    nmin = np.floor(np.min(frac_coords_sym, axis=0)) - maxr
    nmax = np.ceil(np.max(frac_coords_sym, axis=0)) + maxr
    all_ranges = [np.arange(x, y) for x, y in zip(nmin, nmax)]
    neighbors = [[] for _ in range(len(frac_coords))]
    all_fcoords = np.mod(frac_coords_sym, 1)
    coords_in_cell = np.dot(all_fcoords, lattice_mat)
    #site_coords = np.dot(self.frac_coords_sym, lattice_mat)
    all_indices = np.arange(len(frac_coords_sym))  # with summetric positions
    indices = np.arange(len(frac_coords))
    for image in itertools.product(*all_ranges):
        coords = np.dot(frac_coords, lattice_mat)
        z = (coords_in_cell[:, None, :] - coords[None, :, :] + np.dot(image, lattice_mat))
        all_dists = np.sum(z ** 2, axis=-1) ** 0.5
        all_within_r = np.bitwise_and(all_dists <= r, all_dists > 1e-1)
        for (j, d, diff, within_r) in zip(all_indices, all_dists, z, all_within_r):
            for i in indices[within_r]:
                if d[i] > bond_tol:
                    neighbors[i].append([j % n_atoms, diff[i], j // n_atoms])
    return np.array(neighbors, dtype="object")

def is_edge_added(added_edges, edge_i, edge_j, op_id, diff):
    if not (edge_i, edge_j, op_id) in added_edges:
        return False
    edges_dist = np.array(added_edges[(edge_i, edge_j, op_id)]) - np.expand_dims(diff, axis=0)
    edges_dist = np.min(np.max(np.abs(edges_dist), axis=1), axis=0)
    return edges_dist < 1e-4

def nearest_neighbor_edges(
        atoms,
        operations,
        cutoff,
        n_neighbours,
        dtype
    ):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbours = get_all_neighbors(atoms, operations, r=cutoff)
    min_nbrs = min(len(neighbourlist) for neighbourlist in all_neighbours)
    
    while min_nbrs < n_neighbours:
        #logging.info("Increasing cutoff")
        lat = atoms.lattice
        cutoff += 1
        all_neighbours = get_all_neighbors(atoms, operations, r=cutoff)
        min_nbrs = min(len(neighbourlist) for neighbourlist in all_neighbours)
    
    n_atoms = len(atoms.elements)
    inverse_operations = inverse_operations_index(operations)
    u = []
    v = []
    r = []
    sym = []
    added_edges = {}
    lattice, inv_lattice = atoms.lattice_mat.T, np.linalg.inv(atoms.lattice_mat.T)
    for site_id, neighborlist in enumerate(all_neighbours):
        neighborlist = np.array(sorted(neighborlist, key=lambda x: (x[1]**2).sum()), dtype=object)
        ids, diffs, op_id = neighborlist.T  # unpack columns
        ids, diffs, op_id = ids.astype(int), np.array(list(diffs)), op_id.astype(int)
        symmetric = (site_id == ids) & (op_id != 0)
        for neighbor_i in range(n_neighbours):
            neig_id, neig_op_id, neig_diff = ids[neighbor_i], op_id[neighbor_i], diffs[neighbor_i]
            inv_op_id = inverse_operations[neig_op_id]
            inv_op_matrice = operations[inv_op_id][0]
            if type(inv_op_matrice) is torch.Tensor:
                inv_op_matrice = inv_op_matrice.cpu().numpy()
            inv_diff = -lattice@inv_op_matrice@inv_lattice@neig_diff
            assert np.abs(np.linalg.norm(inv_diff) - np.linalg.norm(neig_diff)) < 2e-2, "Inconsistent lattice / symmetry operations"
            if not is_edge_added(added_edges, site_id, neig_id, neig_op_id, neig_diff):
                v.append(neig_id)
                u.append(site_id)
                r.append(neig_diff)
                sym.append(symmetric[neighbor_i])
                if not (site_id, neig_id, neig_op_id) in added_edges:
                    added_edges[site_id, neig_id, neig_op_id] = []
                added_edges[site_id, neig_id, neig_op_id].append(neig_diff)
                if not symmetric[neighbor_i]:
                    v.append(site_id)
                    u.append(neig_id)
                    r.append(inv_diff)
                    sym.append(symmetric[neighbor_i])
                    if not (neig_id, site_id, inv_op_id) in added_edges:
                        added_edges[neig_id, site_id, inv_op_id] = []
                    added_edges[neig_id, site_id, inv_op_id].append(inv_diff)
            else:
                assert is_edge_added(added_edges, neig_id, site_id, inv_op_id, inv_diff), "failed to symmetrise"
    u = torch.tensor(np.array(u), dtype=torch.int)
    v = torch.tensor(np.array(v), dtype=torch.int)
    r = torch.tensor(np.array(r), dtype=dtype)
    sym = torch.tensor(np.array(sym), dtype=torch.int)
    return u, v, r, sym

def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}

def atom_dgl_multigraph(
    atoms,
    operations=[(np.eye(3), np.zeros(3))],  # symmetry operations x -> Ax + b, list of tuples of A and b; first operation should always be identity
    start_cutoff=5.5,
    n_neighbours=80,
    dtype=torch.float
):
    """Obtain a DGLGraph for Atoms object."""
    u, v, r, sym = nearest_neighbor_edges(
        operations=operations,
        atoms=atoms,
        cutoff=start_cutoff,
        n_neighbours=n_neighbours,
        dtype=dtype
    )
    g = dgl.graph((u, v))
    g.ndata["Z"] = torch.tensor(atoms.atomic_numbers)
    g.edata["r"] = r
    g.edata["d"] = r.norm(dim=1)
    g.edata["equiv"] = sym

    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)

    return g, lg


