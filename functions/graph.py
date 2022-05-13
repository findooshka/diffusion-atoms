from jarvis.core.graphs import build_undirected_edgedata, compute_bond_cosines, canonize_edge, get_node_attributes, Graph
from collections import defaultdict
from typing import Optional
from jarvis.core.atoms import Atoms
import numpy as np
import torch
import dgl
from random import sample

def nearest_neighbor_edges(
        atoms=None,
        cutoff=8,
        min_neighbours=12,
        random_neighbours=10,
        id=None,
        use_canonize=False,
    ):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    if min_nbrs < min_neighbours + random_neighbours:
        lat = atoms.lattice
        cutoff += 1
        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=cutoff,
            min_neighbours=min_neighbours,
            random_neighbours=random_neighbours,
            id=id,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[min_neighbours - 1]
        
        chosen = list(range(len(distances)))
        chosen = chosen[:min_neighbours] + sample(chosen[min_neighbours:], random_neighbours)
        
        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[chosen]
        images = images[chosen]
        distances = distances[chosen]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges

class MyGraph(Graph):
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=6.0,
        min_neighbours=12,
        random_neighbours=10,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = False,
    ):
        """Obtain a DGLGraph for Atoms object."""
        if neighbor_strategy == "k-nearest":
            edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=cutoff,
                min_neighbours=min_neighbours,
                random_neighbours=random_neighbours,
                id=id,
                use_canonize=use_canonize,
            )
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        # elif neighbor_strategy == "voronoi":
        #    edges = voronoi_edges(structure)

        u, v, r = build_undirected_edgedata(atoms, edges)

        # build up atom attribute tensor
        sps_features = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            # if include_prdf_angles:
            #    feat=feat+list(prdf[ii])+list(adf[ii])
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features
        g.edata["r"] = r

        if compute_line_graph:
            # construct atomistic line graph
            # (nodes are bonds, edges are bond pairs)
            # and add bond angle cosines as edge features
            lg = g.line_graph(shared=True)
            lg.apply_edges(compute_bond_cosines)
            return g, lg
        else:
            return g
    
