import torch
import numpy as np
import sympy as sb
import logging
from sympy.utilities.lambdify import lambdify
from functions.lattice import get_mask, get_lattice_system

def acotan_sb(x):
    return 0.5*sb.pi - sb.atan(x)

def acotan_np(x):
    return 0.5*np.pi - np.arctan(x)

def softplus_sb(x, t=10.):
    return sb.Piecewise((sb.log(1 + sb.exp(x)), x < t), (x, x >= t))
    
def get_variables(lattice_system):
    if lattice_system == 'triclinic':
        a, b, c, alpha, beta, gamma = sb.symbols('a b c alpha beta gamma', real=True)
    elif lattice_system == 'monoclinic':
        a, b, c, gamma = sb.symbols('a b c gamma', real=True)
        alpha, beta = 2 * [0.]
    elif lattice_system == 'orthorhombic':
        a, b, c = sb.symbols('a b c', real=True)
        alpha, beta, gamma = 3 * [0.]
    elif lattice_system == 'tetragonal':
        a, c = sb.symbols('a c', real=True)
        b = a
        alpha, beta, gamma = 3 * [0.]
    elif lattice_system == 'hexagonal':
        a, c = sb.symbols('a c', real=True)
        b = a
        alpha, beta = 2 * [0.]
        gamma = -1
    elif lattice_system == 'cubic':
        a = sb.symbols('a', real=True)
        b, c = a, a
        alpha, beta, gamma = 3 * [0.]
    else:
        raise ValueError(f"Invalid lattice system: {lattice_system}")
    return (a, b, c, alpha, beta, gamma)
    
def get_symbolic_distance(big_min, lattice_system):
    b6 = get_variables(lattice_system)
    a, b, c, alpha, beta, gamma = b6
    a, b, c = softplus_sb(a), softplus_sb(b), softplus_sb(c)
    #if lattice_system == 'rhombohedral':
    #    alpha = 2 * acotan_sb(alpha) / 3
    #    beta, gamma = alpha, alpha
    #else:
    alpha, beta = acotan_sb(alpha), acotan_sb(beta)
    if big_min:
        max_gamma = 2*sb.pi - alpha - beta
    else:
        max_gamma = alpha + beta
    gamma = acotan_sb(gamma) / sb.pi * (max_gamma - sb.Abs(alpha-beta)) + sb.Abs(alpha-beta)
    v1, v2, v3 = sb.symbols('v_1 v_2 v_3', real=True)
    l_32 = (sb.cos(alpha) - sb.cos(beta)*sb.cos(gamma)) / sb.sin(gamma)
    l = sb.Matrix([[a, 0, 0],
                   [b*sb.cos(gamma), b*sb.sin(gamma), 0],
                   [c*sb.cos(beta), c*l_32, c*sb.sqrt(sb.sin(beta)**2 - l_32**2)]])
    e = sb.Matrix([[v1, v2, v3]])
    e2 = (e@l)
    d = sb.sqrt(e2.dot(e2))
    return l, d, b6, (v1, v2, v3)
    
def get_grad(d, b6):
    grad = sb.Matrix([sb.diff(d, v) for v in b6])
    return grad

def get_hessian(d, b6):
    # not used
    H = sb.Matrix([[sb.diff(sb.diff(d, v1), v2) for v1 in b6] for v2 in b6])
    return H

def substitute(expr, b6, b6_vals, v, v_vals):
    expr = expr.subs(zip(b6, b6_vals))
    expr = expr.subs(zip(v, v_vals))
    return expr

def get_lambda_gradient_func(big_min, lattice_system):
    l, distance, b6, v = get_symbolic_distance(big_min, lattice_system)
    mask = get_mask(lattice_system)
    b6 = [b6[i] for i in mask]
    grad = get_grad(distance, b6)
    return sb.utilities.lambdify([b6, v], grad, modules = [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])

def get_gradient_func():
    logging.info("Preparing lattice gradient functions...")
    lattice_systems = {#'rhombohedral', 
                       'triclinic', 'monoclinic',
                       'orthorhombic', 'tetragonal', 'hexagonal',
                       'cubic'}
    funcs = {}
    for system in lattice_systems:
        funcs[system] = {'big_min': get_lambda_gradient_func(True, system),
                         'small_min': get_lambda_gradient_func(False, system)}
    def result(b6, v, space_group):
        lattice_system = get_lattice_system(space_group)
        mask = get_mask(lattice_system)
        b6_masked = [b6[i] for i in mask]
        if acotan_np(b6[3]) + acotan_np(b6[4]) < np.pi:
            return funcs[lattice_system]['small_min'](b6_masked, v)
        return funcs[lattice_system]['big_min'](b6_masked, v)
    logging.info("...Done")
    return result

