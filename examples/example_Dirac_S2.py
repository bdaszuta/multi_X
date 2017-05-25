#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Thu May 25 2017> @ 12:45:00

@author: Boris Daszuta
@function: Simple example of evolving Dirac on S2.
"""
from numpy import (abs, array, dot, sqrt, zeros)
import matplotlib.pyplot as plt

import multi_SWSH as ms


def calculate_charge_Q(psi_lm):
    '''
    Calculate the charge 'Q' associated with the coefficients of the Dirac
    field.
    Q = \int |\psi_-|^2 + |\psi_+|^2 d\Omega
    '''
    return abs(dot(psi_lm[0, :].conj(), psi_lm[0, :]) +
               dot(psi_lm[1, :], psi_lm[1, :].conj()))


def sys_Dirac_S2(t, psi_lm, **kwargs):
    '''
    Evaluate the RHS of the Dirac EOM.

    Parameters
    ----------
    t : float
        Current time.

    psi_lm : array-like
        Current field configuration.

    **kwargs
        Keys 'mu' and 'L' should be passed.
    '''
    # extract parameters
    mu, L = kwargs['mu'], kwargs['L']

    # build (lookup) coefficient space representation of operators.
    eth_m = ms.eth_build(s=+1, L=L, type=-1, is_dense=True,
                         is_half_integer=True)
    eth_p = ms.eth_build(s=-1, L=L, type=+1, is_dense=True,
                         is_half_integer=True)

    # apply eth operators
    eth_psi_lm = ms.eth_apply(salm=psi_lm, eth_lm=array([eth_m, eth_p]),
                              is_dense=True, is_half_integer=True)

    # build system rhs
    return array([-1j * mu * psi_lm[0, :] - eth_psi_lm[1, :],
                  +1j * mu * psi_lm[1, :] - eth_psi_lm[0, :]])


def sys_Dirac_S2_post_step_fcn(t, psi_lm, **kwargs):
    '''
    Function to call after each Runge-Kutta step. Here we calculate the charge
    Q.
    '''
    t_idx = kwargs['t_idx']
    if t_idx % kwargs['t_idx_key'] == 0:
        ix = t_idx // kwargs['t_idx_key']
        kwargs['Q_arr'][ix] = calculate_charge_Q(
            psi_lm)
        kwargs['t_arr'][ix] = t
    elif t_idx == (kwargs['N_t'] - 1):
        kwargs['Q_arr'][-1] = calculate_charge_Q(psi_lm)
        kwargs['t_arr'][-1] = t


def _ev_step_RK4(t, z, dt, sys_fcn=None, **kwargs):
    dt_2 = dt / 2
    k1 = sys_fcn(t, z, **kwargs)
    k2 = sys_fcn(t + dt_2, z + dt_2 * k1, **kwargs)
    k3 = sys_fcn(t + dt_2, z + dt_2 * k2, **kwargs)
    k4 = sys_fcn(t + dt, z + dt * k3, **kwargs)

    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_RK4(ti, tf, N_t, z_ini, sys_fcn=None, sys_post_step_fcn=None,
               **kwargs):
    '''
    Solve a system using the classic Runge-Kutta 4.
    '''
    dt = (tf - ti) / N_t
    t = ti
    z = z_ini

    for t_idx in range(int(N_t)):
        z = _ev_step_RK4(t, z, dt, sys_fcn=sys_fcn, t_idx=t_idx, **kwargs)
        if sys_post_step_fcn is not None:
            sys_post_step_fcn(t, z, t_idx=t_idx, N_t=N_t, **kwargs)
        t = t + dt
    return z

###############################################################################
# specify parameters and initial data
###############################################################################
L = 33            # band-limit (th=ph)
mu = 1.2          # mass parameter
ti, tf = 0, 100   # initial and final times
t_idx_key = 32    # control frequency of when 'Q' is computed
N_t = 10000       # total number of time-steps (dt inferred)

# generate an initial Dirac field in coefficient space
psi_lm_ini = ms.generate_random_salm(s=array([-1, 1]), L_th=L,
                                     is_half_integer=True)

# normalise based on charge Q (probability)
psi_lm_ini = psi_lm_ini / sqrt(abs(calculate_charge_Q(psi_lm_ini)))

# store time and charge at key time
t_arr = zeros(N_t // t_idx_key + 2)
Q_arr = zeros(N_t // t_idx_key + 2)

# evolve.
psi_lm = evolve_RK4(ti, tf, N_t, psi_lm_ini,
                    sys_fcn=sys_Dirac_S2,
                    sys_post_step_fcn=sys_Dirac_S2_post_step_fcn,
                    mu=mu, L=L, t_idx_key=t_idx_key,
                    t_arr=t_arr, Q_arr=Q_arr)

# inspect Q
plt.semilogy(t_arr, abs(1-Q_arr), 'o-r')
plt.grid(True)
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$|1 - Q|$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

#
# :D
#
