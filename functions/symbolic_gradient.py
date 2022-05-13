import torch
import numpy as np
import sympy as sb
from sympy.utilities.lambdify import lambdify


def acotan_sb(x):
    return 0.5*sb.pi - sb.atan(x)

def acotan_np(x):
    return 0.5*np.pi - np.arctan(x)

def positive(x):
    if x is float:
        x = torch.tensor(x)
    return sb.Piecewise((sb.log(1 + sb.exp(x)), x < 10), (x, x >= 10))

def get_symbolic_distance(big_min):
    a, b, c, alpha, beta, gamma = sb.symbols('a b c alpha beta gamma', real=True)
    b6 = (a, b, c, alpha, beta, gamma)
    a, b, c = positive(a), positive(b), positive(c)
    alpha, beta = acotan_sb(alpha), acotan_sb(beta)
    #max_gamma = sb.Min(alpha + beta, 2*sb.pi - alpha - beta)
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
    d = sb.sqrt(e2.dot(e2))#.simplify()
    #return sb.sqrt(e2.dot(e2))
    return l, d, b6, (v1, v2, v3)

def get_grad(d, b6):
    grad = sb.Matrix([sb.diff(d, v) for v in b6])
    return grad

def get_hessian(d, b6):
    H = sb.Matrix([[sb.diff(sb.diff(d, v1), v2) for v1 in b6] for v2 in b6])
    return H

def substitute(expr, b6, b6_vals, v, v_vals):
    expr = expr.subs(zip(b6, b6_vals))
    expr = expr.subs(zip(v, v_vals))
    return expr

def get_lambda_gradient_func(big_min):
    l, distance, b6, v = get_symbolic_distance(big_min)
    grad = get_grad(distance, b6)
    return sb.utilities.lambdify([b6, v], grad, modules = [{'Heaviside': lambda x: np.heaviside(x, 1)}, 'numpy'])

def get_gradient_func():
    big_min_f, small_min_f = get_lambda_gradient_func(True), get_lambda_gradient_func(False)
    def result(b6, v):
        if acotan_np(b6[3]) + acotan_np(b6[4]) < np.pi:
            return small_min_f(b6, v)
        return big_min_f(b6, v)
    return result

