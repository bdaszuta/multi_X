#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:01:20

@author: Boris Daszuta
@function: Example of exponential decay and finite band-limit representation.

We consider:
CF(r, th) := cos(r) * (4 + 3 * cos(2 * th))

Define:
f0(r, th) := \eth(CF) / (sqrt(2) * r * CF ** 2)

Make the restriction r = 1.
"""
from numpy import (abs, arange, array, complex128, cos, meshgrid, sin, sqrt, zeros)
import matplotlib.pyplot as plt

import multi_X_SWSH as mXs


def plot_decay(L_th=None, L_ph=None):

    # generate lattice
    th_S2, ph_S2 = mXs.build_grid(L_th=L_th, L_ph=L_ph, is_half_integer=False,
                                  is_extended=False)

    # ensure sampling correctly redimensionalises
    th_S2, ph_S2 = meshgrid(th_S2, ph_S2, indexing='ij')

    # define test functions
    CF = lambda r, th: cos(r) * (4 + 3 * cos(2 * th))
    eth_CF = lambda r, th: cos(r) * (-6 * sin(2 * th))

    CF_a = lambda th, ph: CF(1, th)
    CF_b = lambda th, ph: (1 / CF(1, th))**2
    CF_c = lambda th, ph: eth_CF(1, th)

    # specify spin-weights
    s_arr = array([0, 0, 1, 1])

    # evaluate test functions on 2-sphere
    CF_a_N_S2 = array(CF_a(th_S2, ph_S2))
    CF_b_N_S2 = array(CF_b(th_S2, ph_S2))
    CF_c_N_S2 = array(CF_c(th_S2, ph_S2))

    f0_N_S2 = CF_c_N_S2 / (sqrt(2)) * CF_b_N_S2

    # transform to coefficient space
    sf_arr = array([CF_a_N_S2, CF_b_N_S2, CF_c_N_S2, f0_N_S2])
    salm_arr = mXs.sf_to_salm(s_arr, sf_arr, is_half_integer=False)

    # extract m = 0 entries
    num_fcn_r = salm_arr.shape[0]
    salm_arr_m0 = zeros((num_fcn_r, L_th + 1), dtype=complex128)
    l0_idx = mXs.arr_idx_map(l=arange(L_th + 1), m=0,
                             is_half_integer=False)

    for fi in range(4):
        salm_arr_m0[fi, :] = salm_arr[fi, l0_idx]

    plt.figure(1)
    plt.semilogy(abs(salm_arr_m0[0, :]), '.r')
    plt.semilogy(abs(salm_arr_m0[1, :]), '.b')
    plt.semilogy(abs(salm_arr_m0[2, :]), '.g')
    plt.semilogy(abs(salm_arr_m0[3, :]), '.k')

    ax = plt.gca()
    ax.set_xlim(0, L_th)
    plt.tight_layout()
    plt.xlabel(r'$l$', fontsize=12)
    plt.ylabel(r'$|{}_s a_{l,0}|$', fontsize=12)
    plt.tight_layout()

    return salm_arr_m0

plot_decay(L_th=128, L_ph=1)

#
# :D
#
