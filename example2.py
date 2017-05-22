#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:01:20

@author: Boris Daszuta
@function: Example of SWSH transformation for (half) integer spin-weighted
fields ${}_sf$
"""
import multi_SWSH as mS

import numpy as np

# Boolean for caching reduced Wigner elements
cache_del = False


def JoergEmail(L_th, L_ph):
    '''

    '''

    print('=> test for Joerg')
    ###############################################
    # Test for Joerg:
    #
    # Specify the following CF:
    # CF(r,th) = cos(r)*(4+3*cos(2*th))
    #
    # Define the fcn:
    # f0 = \eth(CF)/[sqrt(2)*r CF**2]
    #
    # Now restrict s.t. r=1 and consider the cases:
    #
    ###############################################

    import matplotlib.pyplot as plt

    ########
    # Integer
    ########

    # Indexing map:
    ind_map = lambda li, mi: li * (li + 1) + mi

    # Num. nodes on S^2 grid [must be even]
    # To sample up to l=L /\ m=L take: N_th=2(L+2) & N_ph=2L+2
    #L_th, L_ph = 128, 2
    N_th, N_ph = 2 * (L_th + 2), 2 * L_ph + 2

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Extended domain lattice
    int_th = np.pi / (N_th - 1) * np.arange(0, (N_th_E))
    int_ph = 2 * np.pi / N_ph * np.arange(0, (N_ph - 1) + 1)

    CF = lambda r, th: np.cos(r) * (4 + 3 * np.cos(2 * th))
    eth_CF = lambda r, th: np.cos(r) * (-6 * np.sin(2 * th))

    CF_a = lambda th, ph: CF(1, th)
    CF_b = lambda th, ph: (1 / CF(1, th))**2
    CF_c = lambda th, ph: eth_CF(1, th)

    s_arr = np.array([0, 0, 1, 1])

    # th,ph coords:
    th_S2, ph_S2 = np.meshgrid(int_th[:N_th], int_ph[:N_ph], indexing='ij')
    # Evaluate fcn on 2-sphere
    CF_a_N_S2 = np.array(CF_a(th_S2, ph_S2), dtype=np.complex128)
    CF_b_N_S2 = np.array(CF_b(th_S2, ph_S2), dtype=np.complex128)
    CF_c_N_S2 = np.array(CF_c(th_S2, ph_S2), dtype=np.complex128)

    f0_N_S2 = CF_c_N_S2 / (np.sqrt(2)) * CF_b_N_S2

    sf_arr = np.array([CF_a_N_S2, CF_b_N_S2, CF_c_N_S2, f0_N_S2])
    salm_arr = mS.int_salm(s_arr, sf_arr)

    num_fun_r = salm_arr.shape[0]

    # Extract l=0 entries for plotting
    salm_arr_angular = np.zeros((num_fun_r, L_th + 1), dtype=np.complex128)

    for f in np.arange(num_fun_r):
        for l in np.arange(L_th + 1):
            salm_arr_angular[f, l] = salm_arr[f, ind_map(l, 0)]

    # Eth ops.
    # raising: -sqrt((l-s)(l+s+1))
    # lowering: sqrt((l+s)(l-s+1))

    plt.figure(1)

    plt.semilogy(np.abs(salm_arr_angular[0]), '.r')
    plt.semilogy(np.abs(salm_arr_angular[1]), '.b')
    plt.semilogy(np.abs(salm_arr_angular[2]), '.g')
    plt.semilogy(np.abs(salm_arr_angular[3]), '.k')

    ax = plt.gca()
    ax.set_xlim(0, 128)
    plt.tight_layout()
    plt.xlabel(r'$l$', fontsize=12)
    plt.ylabel(r'$|{}_s a_{l,0}|$', fontsize=12)
    plt.tight_layout()

    return salm_arr_angular



#
# :D
#
