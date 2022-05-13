import torch
from jarvis.core.atoms import Atoms
import numpy as np

from functions.atomic import get_output, return_to_lattice, add_random_atoms, get_L_inv_estimate, B6_transform, B6_inv_transform, B9_to_B6, B6_to_B9

    


def lattice_loss(model,
                 atoms,
                 t: torch.LongTensor,
                 e: torch.Tensor,
                 b: torch.Tensor,
                 device,
                 lengths_mult,
                 angles_mult,
                 gradient):
    ## gradient: function from basis6, edge vector to gradient of length wrt basis6
    x0 = torch.tensor(atoms.cart_coords, device=e.device)
    assert len(x0.shape) == 2
    #s_t = b.cumsum(dim=0).index_select(0, t)
    s_t = b.index_select(0, t)
    x = x0 + e * s_t.sqrt()
    x_numpy = x.cpu().detach().numpy()
    
    x_numpy = return_to_lattice(x_numpy, atoms.lattice_mat)
    noised_atoms = Atoms(lattice_mat=atoms.lattice_mat,
                         coords=x_numpy,
                         elements=atoms.elements,
                         cartesian=True)
    
    lattice = torch.tensor(atoms.lattice_mat, device=device)
    lattice_neg = lattice.clone()
    lattice_neg[0] = -lattice_neg[0]
    lattice_neg = lattice_neg.detach()
    lattice6 = B9_to_B6(torch.tensor(atoms.lattice_mat, device=device), device=device)
    lattice6 = B6_transform(lattice6)
    lattice_noise = torch.normal(torch.zeros(6, device=device))
    lattice_noise_scaled = torch.tensor(3*[lengths_mult] + 3*[angles_mult], device=device) * lattice_noise * s_t.sqrt()
    noised_lattice6 = lattice6 + lattice_noise
    noised_transformed_lattice6 = noised_lattice6.clone()
    noised_lattice6 = B6_inv_transform(noised_lattice6)
    noised_lattice9 = B6_to_B9(noised_lattice6, so=torch.linalg.det(lattice), device=device)
    if noised_lattice9 is None or torch.abs(torch.linalg.det(noised_lattice9)) < 1. * len(atoms.elements):
        return None
    
    #print(torch.linalg.det(noised_lattice9), torch.linalg.det(lattice))
    #print(torch.linalg.det(noised_lattice9) * torch.linalg.det(lattice) < 0)   
    
    noised_atoms = Atoms(lattice_mat=noised_lattice9.detach().cpu(),
                         coords=noised_atoms.frac_coords,
                         elements=atoms.elements,
                         cartesian=False)
    
    noise_estimate = get_output(noised_atoms, model, t, device, output_type="lattice", b6=noised_transformed_lattice6, gradient=gradient, emax=50)[2]
    #noise_estimate = torch.zeros(6, device=device)
                        
    #L_estimate = get_output(noised_atoms, model, t, device, output_type="lattice", gradient=gradient)[2]
    #if L_estimate == None:
    #    print("skip")
    #    return None
    #L_estimate = L_estimate.to(dtype=noised_lattice9.dtype)
    #lattice_estimate = torch.matmul(noised_lattice9, L_estimate)
    #lattice_estimate6 = B9_to_B6(lattice_estimate, device=device)
    #B6_transform(lattice_estimate6)
    return (noise_estimate - lattice_noise).square().mean()
    

def noise_estimation_loss(model,
                          atoms,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          device):
    x0 = torch.from_numpy(atoms.cart_coords).to(device=device)
    assert len(x0.shape) == 2
    #a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1)
    #assert a.shape == (1, 1)
    #x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    
    #s_t = b.cumsum(dim=0).index_select(0, t).view(-1, 1)
    s_t = b.index_select(0, t).view(-1, 1)
    x = x0 + e * s_t.sqrt()
    
    x_numpy = x.cpu().detach().numpy()
            
    x_numpy = return_to_lattice(x_numpy, atoms.lattice_mat)
    noised_atoms = Atoms(lattice_mat=atoms.lattice_mat,
                         coords=x_numpy,
                         elements=atoms.elements,
                         cartesian=True)
    
    if False:
        noised_atoms, fake = add_random_atoms(noised_atoms, add_atoms_l)
        pos_eps, fake_estimate, lattice_eps = get_output(noised_atoms, model, t, device, output_type="atoms")
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(fake_estimate, torch.tensor(fake, device=device).view(-1, 1))
        loss *= count_loss_coef
    else:
        pos_eps, fake_estimate, lattice_eps = get_output(noised_atoms, model, t, device, output_type="edges")
        #loss = crystal_loss(e, pos_eps, torch.from_numpy(atoms.lattice_mat).to(device).float(), s_t, device)
        loss = (e - pos_eps).square().mean()
    
    return loss
