"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from models.alignn_utils import RBFExpansion, BaseSettings

import math

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 3
    gcn_layers: int = 3
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 96
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    output_atom_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    #link: Literal["identity", "log", "logit"] = "identity"
    #zero_inflated: bool = False
    
    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layer(x)
        

class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, time_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        
        self.time_proj = nn.Linear(time_features, output_features)
        
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.layer_norm_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.layer_norm_nodes = nn.LayerNorm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        time_feats: torch.Tensor
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()
        
        time_feats = self.time_proj(time_feats)

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        #g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_src"] = self.src_gate(node_feats) + time_feats
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.layer_norm_nodes(x))
        y = F.silu(self.layer_norm_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        time_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features, time_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features, time_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y, t)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z, t)

        return x, y, z


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        
        self.timestep_embedding_features = config.embedding_features
        
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        
        self.timestep_embedding = nn.Sequential(
            MLPLayer(self.timestep_embedding_features, config.hidden_features),
            MLPLayer(config.hidden_features, config.hidden_features)
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        #self.edges_layer1 = ALIGNNConv(config.hidden_features, config.hidden_features, config.hidden_features)
        self.edges_layer1 = EdgeGatedGraphConv(config.hidden_features, config.hidden_features, config.hidden_features)
        self.edges_layer2 = EdgeGatedGraphConv(config.hidden_features, config.hidden_features, config.hidden_features)
        self.edges_readout = nn.Linear(config.hidden_features, config.output_features)
        
        self.atoms_layer = EdgeGatedGraphConv(config.hidden_features, config.hidden_features, config.hidden_features)
        self.atoms_readout = nn.Linear(config.hidden_features, config.output_features)
        
        self.lattice_layer = ALIGNNConv(config.hidden_features, config.hidden_features, config.hidden_features)
        self.lattice_readout = nn.Linear(config.hidden_features, config.output_features)
    
    def lattice_projection(self,
                           lattice: torch.Tensor,
                           r: torch.Tensor,
                           data: torch.Tensor):
        r = r / torch.linalg.norm(r, dim=1, keepdim=True)
        lattice = lattice / torch.linalg.norm(lattice, dim=1, keepdim=True)
        cos = torch.einsum('ik, jk->ij', r, lattice)
        return torch.einsum('ik,ij->ikj', cos, data).flatten(start_dim=1)
    
    def lattice_output(self, g, lg, x, y, z, t, r, lattice):
        # lattice noise estimation
        
        y_lattice = self.lattice_redim_edges(y)
        y_lattice = self.lattice_projection(lattice, r, y_lattice)
        y_lattice = self.lattice_redim2_edges(y_lattice)
        
        x, y_lattice, z = self.lattice_layer(g, lg, x, y_lattice, z, t)
        x, y_lattice = self.lattice_layer2(g, x, y_lattice, t)
        x, y_lattice = self.lattice_layer3(g, x, y_lattice, t)
        
        y_lattice = self.lattice_readout(y_lattice)
        
        result = 15 * torch.tanh(y_lattice.mean(dim=0)/15)
        return result

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        lattice: torch.Tensor,
        t: torch.Tensor,
        output_type: str
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        
        t = get_timestep_embedding(t, self.timestep_embedding_features)
        t = self.timestep_embedding(t)
        
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()
        
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        r = g.edata.pop("r")
        bondlength = torch.norm(r, dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z, t)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y, t)

        edges_output, atoms_output, lattice_output = None, None, None
        

        if output_type == "all" or output_type == "edges":
            x_edges, y_edges = self.edges_layer1(g, x, y, t)
            x_edges, y_edges = self.edges_layer2(g, x_edges, y_edges, t)
            edges_output = self.edges_readout(y_edges)
        
        if output_type == "all" or output_type == "atoms":
            x_atoms, _ = self.atoms_layer(g, x, y, t)
            atoms_output = self.atoms_readout(x_atoms)
        
        if output_type == "all" or output_type == "lattice":
            x_lattice, y_lattice, z_lattice = self.lattice_layer(g, lg, x, y, z, t)
            lattice_output = self.lattice_readout(y_lattice)
            #lattice_output = self.lattice_output(g, lg, x, y, z, t, r, lattice)
        
        
        return edges_output, atoms_output, lattice_output
