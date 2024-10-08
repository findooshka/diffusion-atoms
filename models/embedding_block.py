import numpy as np
import torch
import torch.nn as nn

from models.envelope import Envelope
from models.initializers import GlorotOrthogonal

class EmbeddingBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 num_radial,
                 bessel_funcs,
                 cutoff,
                 envelope_exponent,
                 num_atom_types=96,
                 step_count=20,
                 activation=None):
        super(EmbeddingBlock, self).__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.step_embedding = nn.Embedding(step_count, emb_size)
        self.equiv_embedding = nn.Embedding(2, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        #self.dense_gradient = nn.Linear(6, emb_size)
        self.dense = nn.Linear(emb_size * 5, emb_size)
        self.reset_params()
    
    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense.weight)

    def edge_init(self, edges):
        """ msg emb init """
        # m init
        rbf = self.dense_rbf(edges.data['rbf'])
        if self.activation is not None:
            rbf = self.activation(rbf)
        
        equiv_emb = self.equiv_embedding(edges.data['equiv'].squeeze(-1))
        #gradient = self.dense_gradient(edges.data['gradient_mat'])
        
        #m = torch.cat([edges.src['h'], edges.dst['h'], rbf, self.step_embedding(edges.src['step']), equiv_emb, gradient], dim=-1)
        m = torch.cat([edges.src['h'], edges.dst['h'], rbf, self.step_embedding(edges.src['step']), equiv_emb], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)
        
        # rbf_env init
        d_scaled = edges.data['d'] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {'m': m, 'rbf_env': rbf_env}

    def forward(self, g):
        g.ndata['h'] = self.embedding(g.ndata['Z'])
        g.apply_edges(self.edge_init)
        return g
