import torch.nn as nn

from models.initializers import GlorotOrthogonal

class ResidualLayer(nn.Module):
    def __init__(self, units, activation=None):
        super(ResidualLayer, self).__init__()

        self.activation = activation
        
        #self.lnorm_1 = nn.LayerNorm(normalized_shape=units)
        #self.lnorm_2 = nn.LayerNorm(normalized_shape=units)
        
        self.dense_1 = nn.Linear(units, units)
        self.dense_2 = nn.Linear(units, units)
    
        self.reset_params()
    
    def reset_params(self):
        GlorotOrthogonal(self.dense_1.weight)
        nn.init.zeros_(self.dense_1.bias)
        GlorotOrthogonal(self.dense_2.weight)
        nn.init.zeros_(self.dense_2.bias)

    def forward(self, inputs):
        x = self.dense_1(inputs)
        #x = self.lnorm_1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        #x = self.lnorm_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x
