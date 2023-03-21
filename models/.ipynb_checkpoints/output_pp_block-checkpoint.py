import torch.nn as nn
import torch
import dgl
import dgl.function as fn

from models.initializers import GlorotOrthogonal

class OutputPPBlock(nn.Module):
    def __init__(self,
                 emb_size,
                 out_emb_size,
                 num_radial,
                 num_dense,
                 activation=None,
                 output_init=nn.init.zeros_,
                 extensive=True,
                 n_atoms=95):
        super(OutputPPBlock, self).__init__()

        self.activation = activation
        self.output_init = output_init
        self.extensive = extensive
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection = nn.Linear(emb_size, out_emb_size, bias=False)
        self.layer_norm = nn.LayerNorm(normalized_shape=out_emb_size)
        self.dense_layers = nn.ModuleList([
            nn.Linear(out_emb_size, out_emb_size) for _ in range(num_dense)
        ])
        self.dense_final = nn.Linear(out_emb_size, 2, bias=False)
        
        self.dense_rbf_nodes = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection_nodes = nn.Linear(emb_size, out_emb_size, bias=False)
        self.layer_norm_nodes = nn.LayerNorm(normalized_shape=out_emb_size)
        self.dense_layers_nodes = nn.ModuleList([
            nn.Linear(out_emb_size, out_emb_size) for _ in range(num_dense)
        ])
        self.dense_final_nodes = nn.Linear(out_emb_size, n_atoms, bias=False)
        self.reset_params()
    
    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf_nodes.weight)
        GlorotOrthogonal(self.up_projection_nodes.weight)
        for layer in self.dense_layers_nodes:
            GlorotOrthogonal(layer.weight)
        self.output_init(self.dense_final_nodes.weight)
        
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.up_projection.weight)
        for layer in self.dense_layers:
            GlorotOrthogonal(layer.weight)
        self.output_init(self.dense_final.weight)

    def forward(self, g):
        with g.local_scope():
            g.edata['tmp'] = g.edata['m'] * self.dense_rbf(g.edata['rbf'])
            g.edata['tmp'] = self.up_projection(g.edata['tmp'])
            g.edata['tmp'] = self.layer_norm(g.edata['tmp'])
            for layer in self.dense_layers:
                g.edata['tmp'] = layer(g.edata['tmp'])
                if self.activation is not None:
                    g.edata['tmp'] = self.activation(g.edata['tmp'])
            g.edata['tmp'] = self.dense_final(g.edata['tmp'])
            
            g.edata['tmp_nodes'] = g.edata['m'] * self.dense_rbf_nodes(g.edata['rbf'])
            g_reverse = dgl.reverse(g, copy_edata=True)
            g_reverse.update_all(fn.copy_e('tmp_nodes', 'x'), fn.sum('x', 'tmp_nodes'))
            g.ndata['tmp_nodes'] = self.up_projection_nodes(g_reverse.ndata['tmp_nodes'])
            g.ndata['tmp_nodes'] = self.layer_norm_nodes(g.ndata['tmp_nodes'])
            for layer in self.dense_layers_nodes:
                g.ndata['tmp_nodes'] = layer(g.ndata['tmp_nodes'])
                if self.activation is not None:
                    g.ndata['tmp_nodes'] = self.activation(g.ndata['tmp_nodes'])
            g.ndata['tmp_nodes'] = self.dense_final_nodes(g.ndata['tmp_nodes'])
            
            return [g.edata['tmp'], g.ndata['tmp_nodes']]
