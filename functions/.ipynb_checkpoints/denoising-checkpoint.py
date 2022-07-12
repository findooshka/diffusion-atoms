import torch
from functions.atomic import return_to_lattice, get_output, B6_transform, B6_inv_transform, get_sampling_coords, B9_to_B6, B6_to_B9
from jarvis.core.atoms import Atoms
import numpy as np
from collections import Counter

def compute_st(b, t):
    return b.cumsum(dim=0).index_select(0, t.long()).view(-1, 1)

def remove_atom(atoms, fake_probabilities, mult, softmax_l=20.):
    fake_softmax = np.exp(softmax_l*fake_probabilities)
    fake_softmax = np.ravel(fake_softmax) / fake_softmax.sum()
    #print(fake_softmax)
    remove_i = np.argmax(np.random.multinomial(1, fake_softmax))
    #remove_i = np.argmax(fake_probabilities)
    if len(atoms.elements) > 1 and np.random.rand() < mult * fake_probabilities[remove_i]:
        x = atoms.cart_coords
        x = np.delete(x, remove_i, 0)
        elements = atoms.elements.copy()
        print("removed", elements.pop(remove_i))
        return Atoms(coords = x,
                     lattice_mat = atoms.lattice_mat,
                     elements = elements,
                     cartesian = True)
    return atoms
    
def lattice_step(atoms, sigma, step, t, model, device, gradient, lengths_mult, angles_mult, l=0.1):
    lattice = B6_transform(B9_to_B6(torch.tensor(atoms.lattice_mat)))
    score = None
    while score == None:
        score = get_output(atoms, model, t, device, output_type='lattice', b6=lattice, gradient=gradient, emax=200)[1].cpu()
    score /= sigma.item()**0.5
    score[:3] /= lengths_mult
    score[3:] /= angles_mult
    z = np.random.normal(size=(6,))
    noise = step**0.5 * z
    penalty = np.zeros(6)
    penalty[3:] = -l * step**0.5 * lattice[3:] * np.abs(lattice[3:])
    print(sigma, lattice, noise, -0.5*step*score, penalty)
    lattice = lattice - 0.5*step*score + noise + penalty
    #lattice[:3] = torch.maximum(lattice[:3], torch.tensor(2))
    lattice9 = B6_to_B9(B6_inv_transform(lattice), 'cpu', np.linalg.det(atoms.lattice_mat)).cpu().detach().numpy()
    randomized_atoms = Atoms(coords = atoms.frac_coords,
                             lattice_mat = lattice9,
                             elements = atoms.elements,
                             cartesian = False)
    print("det", np.linalg.det(atoms.lattice_mat))
    return randomized_atoms, lattice9, randomized_atoms.cart_coords

def positions_step(xt, atoms, sigma, step, t, model, device):
    z = np.random.normal(size=xt.shape)
    score = get_output(atoms, model, t, device, output_type='edges')[0] / sigma**0.5        
    score = score.cpu().detach().numpy()
    xt = xt - 0.5*step*score + 1e-3 * step**0.5 * z
    xt = return_to_lattice(xt, atoms.lattice_mat)
    atoms = Atoms(coords = xt,
                  lattice_mat = atoms.lattice_mat,
                  elements = atoms.elements,
                  cartesian = True)
    return atoms, xt

@torch.no_grad()
def langevin_dynamics(atoms, b, model, device, gradient, lengths_mult, angles_mult, T=60, epsilon=1e-4, noise_original_positions=False):
    b_iter = reversed(list(enumerate(b)))
    lattice = (3 + 2.* np.abs(np.random.normal(size=(3,)))) * np.eye(3)
    #lattice = atoms.lattice_mat
    atoms = Atoms(coords = atoms.frac_coords,
                  lattice_mat = lattice,
                  elements = atoms.elements,
                  cartesian = False)
    xt = get_sampling_coords(atoms, noise_original_positions, 0.15)
    #xt = atoms.frac_coords @ lattice
    result = [atoms]
    for t, sigma in b_iter:
        step = epsilon * (sigma / b[0])
        step = step.cpu().detach().numpy()
        print("Time: {}, Step size: {}".format(t, step))
        randomized_atoms = Atoms(coords = xt,
                                 lattice_mat = lattice,
                                 elements = atoms.elements,
                                 cartesian = True)
        result.append(randomized_atoms)
        for i in range(T):
            t_tensor = torch.ones(1).to(device)*t
            #if 5 > t:
            randomized_atoms, xt = positions_step(xt, randomized_atoms, sigma, step, t_tensor, model, device)
            
            if 10 > t:
                randomized_atoms, lattice, xt = lattice_step(randomized_atoms, sigma, step, t_tensor, model, device, gradient=gradient, lengths_mult=lengths_mult, angles_mult=angles_mult)
    result.append(randomized_atoms)
    return result, None
            

def generalized_steps(x, atoms, seq, model, b, device, remove_atoms, remove_atoms_mult, max_t, lattice_noise, lattice_noise_mult, **kwargs):
    with torch.no_grad():
        #n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        atoms_seq = [atoms]
        #xs = [atoms.cart_coords]
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            atoms = atoms_seq[-1]
            t = (torch.ones(1) * i).to(device)
            next_t = (torch.ones(1) * j).to(device)
            st = compute_st(b, t)
            xt = xs[-1]
            xt = return_to_lattice(xt, atoms_seq[-1].lattice_mat)
            randomized_atoms = Atoms(coords = xt,
                                     lattice_mat = atoms_seq[-1].lattice_mat,
                                     elements = atoms.elements,
                                     cartesian = True)
            if t < max_t:
                et, _, _ = get_output(randomized_atoms, model, t, device, output_type='edges')
            else:
                et, _, _ = get_output(randomized_atoms, model, (torch.ones(1) * max_t).to(device), device)
            #fake = torch.special.expit(fake)
            #fake = fake.cpu().detach().numpy()
            
            step = b.index_select(0, t.long()).cpu().detach().numpy()
            
            x0_t = xt - (et * st.sqrt()).cpu().detach().numpy()
            x0_preds.append(x0_t)
            xt_next = xt - step * et.cpu().detach().numpy() + np.sqrt(step) * np.random.normal(size=xt.shape)
            xs.append(xt_next)
            #xs.append(xt)
            
            
            randomized_atoms = Atoms(coords = xs[-1],
                                     lattice_mat = atoms_seq[-1].lattice_mat,
                                     elements = atoms.elements,
                                     cartesian = True)
            if lattice_noise:
                L = None
                fails = 0
                while L is None and fails < 5:
                    _, _, L = get_output(randomized_atoms, model, t, device, output_type='lattice', s_t=st)
                    fails += 1
                if fails == 5:
                    break
                L = L.cpu().detach().numpy()
                print(L)
                lattice = atoms.lattice_mat
                predicted_lattice = atoms.lattice_mat @ L
                lattice6 = B9_to_B6(torch.tensor(lattice), 'cpu')
                predicted_lattice6 = B9_to_B6(torch.tensor(predicted_lattice), 'cpu')
                new_lattice6 = lattice6
                #new_lattice6[3:] += step.cpu() * (predicted_lattice6[3:] - lattice6[3:])
                new_lattice6 += b.index_select(0, t.long()).cpu() * (predicted_lattice6 - lattice6)
                new_lattice9 = B6_to_B9(new_lattice6, 'cpu', np.linalg.det(lattice))
                new_lattice = new_lattice9.cpu().detach().numpy()
                print(lattice6, new_lattice6)
                if new_lattice9 is not None:
                    randomized_atoms = Atoms(coords = randomized_atoms.frac_coords,
                                             lattice_mat = new_lattice,
                                             elements = atoms.elements,
                                             cartesian = False)
            #if remove_atoms and i > 200:
            #    atoms_seq[-1] = remove_atom(atoms_seq[-1], fake, remove_atoms_mult)
            #    xs[-1] = atoms_seq[-1].cart_coords
            atoms_seq.append(randomized_atoms)
            xs.append(randomized_atoms.cart_coords)
            if i % 10 == 0:
                print(i,
                      (torch.linalg.norm(et)).cpu().detach().numpy(),
                      (b.index_select(0, t.long())).cpu().detach().numpy(),
                      (st.sqrt()).cpu().detach().numpy(),
                      Counter(atoms_seq[-1].elements),
                      np.linalg.det(atoms_seq[-1].lattice_mat))

    return atoms_seq, x0_preds





def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def ___generalized_steps(x, atoms, seq, model, b, device, **kwargs):
    with torch.no_grad():
        #n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        atoms_seq = [atoms]
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(1) * i).to(device)
            next_t = (torch.ones(1) * j).to(device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            atoms_t = atoms_seq[-1]
            xt = xs[-1]
            xt = return_to_lattice(xt, atoms.lattice_mat)
            randomized_atoms = Atoms(coords = xt,
                                     lattice_mat = atoms.lattice_mat,
                                     elements = atoms.elements,
                                     cartesian = True)
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return atoms_seq, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
