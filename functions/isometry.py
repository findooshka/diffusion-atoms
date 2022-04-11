import torch
import numpy as np
from torch import sin, cos


OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

# not used
def crystal_loss(v1, v2, lattice, s_t, device):
    lattice /= s_t.sqrt()
    difference = (v1 - v2).unsqueeze(2)
    offsets = torch.tensor(OFFSET_LIST, device=device, dtype=lattice.dtype)
    offsets = torch.matmul(offsets, lattice).transpose(0, 1)
    offsets = offsets.unsqueeze(0)
    difference = (difference - offsets).square().min(dim=2)[0]
    return difference.mean()


def B9_to_B6(B9, device):
    # 3x3 matrix -> (a, b, c, alpha, beta, gamma)
    assert B9.shape == (3,3)
    lengths = torch.norm(B9, dim=1)
    result = torch.empty((6,), dtype=B9.dtype, device=device)
    result[[0,1,2]] = lengths
    result[3] = torch.acos(torch.dot(B9[0], B9[1]) / lengths[0] / lengths[1])
    result[4] = torch.acos(torch.dot(B9[0], B9[2]) / lengths[0] / lengths[2])
    result[5] = torch.acos(torch.dot(B9[1], B9[2]) / lengths[1] / lengths[2])
    return result
    
def B6_to_B9(B6, device, so):
    # (a, b, c, alpha, beta, gamma) -> 3x3 matrix
    # if impossible, return None
    # orientation determined by so parameter
    assert B6.shape == (6,)
    if torch.any(B6[:3] <= 0):
        return None
    result = torch.zeros((3,3), dtype=B6.dtype, device=device)
    B6 = B6.flatten()
    result[0, 0] = B6[0]
    result[1, 0] = B6[1] * torch.cos(B6[3])
    result[1, 1] = B6[1] * torch.sin(B6[3])
    result[2, 0] = B6[2] * torch.cos(B6[4])
    result[2, 1] = B6[2] * (torch.cos(B6[5]) - torch.cos(B6[3]) * torch.cos(B6[4])) / torch.sin(B6[3])
    x = B6[2].square() - result[2, 0].square() - result[2, 1].square()
    if x > 0:
        result[2, 2] = torch.sqrt(x)
        if torch.linalg.det(result) * so < 0:
            result[0, 0] = -result[0, 0]
        return result
    #else:
    #    result = torch.zeros((3,3), device=device)
    #    penalty = penalty_mult * (1-x) / B6[2].square()
    return None

class FindBasis(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.alpha = torch.nn.Parameter(2*np.pi*torch.rand((1,1), dtype=dtype, device=device))
        self.beta = torch.nn.Parameter(2*np.pi*torch.rand((1,1), dtype=dtype, device=device))
        self.gamma = torch.nn.Parameter(2*np.pi*torch.rand((1,1), dtype=dtype, device=device))
    
    def forward(self, l):
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        sina, sinb, sing = sin(alpha.clone()), sin(beta.clone()), sin(gamma.clone())
        cosa, cosb, cosg = cos(alpha.clone()), cos(beta.clone()), cos(gamma.clone())
        P = torch.cat([torch.cat([cosa*cosb, cosa*sinb, sina], dim=1),
                       torch.cat([-sina*cosb*cosg - sinb*sing,
                                  -sina*sinb*cosg + cosb*sing,
                                  cosa*cosg], dim=1),
                       torch.cat([sina*cosb*sing - sinb*cosg,
                                  sina*sinb*sing + cosb*cosg,
                                  -cosa*sing], dim=1)])
        return torch.matmul(l, P), P
        
        
def get_isometry(A, B, device, lr=0.001, n=300):
    # find an isometry L, such that BL approximates A
    assert A.shape == (3,3)
    assert B.shape == (3,3)
    A = A.detach()
    B = B.detach()
    
    model = FindBasis(dtype=B.dtype, device=device)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(n):
        optim.zero_grad()
        est = model(B)[0]
        loss = (A - est).square().mean()
        loss.backward(retain_graph=(i!=n-1))
        optim.step()
    print("loss", loss)
        
    return model(B)
