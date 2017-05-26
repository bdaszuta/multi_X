#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:01:20

@author: Boris Daszuta
@function: Example of the action of the eth operators.
"""
from numpy import (array, exp, cos, sin, sqrt, pi, max, abs)
import multi_X_SWSH as ms


def h_int_ex():
    '''
    Half-integer eth test.
    '''
    ###########################################################################
    # some manipulations comparing numerical and analytical operations
    ###########################################################################
    HI_L_th, HI_L_ph = 7, 7

    # use convenience function for construction of numerical grid
    HI_th, HI_ph = ms.build_grid(L_th=HI_L_th, L_ph=HI_L_ph,
                                 is_half_integer=True)
    # broadcast
    (HI_th, HI_ph) = (HI_th[:, None], HI_ph[None, :])

    # sample analytical form of functions:
    # hYa: s=1/2, l=3/2, m=-1/2
    # hYb: s=1/2, l=3/2, m=1/2
    hYa = (exp(-1j * HI_ph / 2) * (1 + 3 * cos(HI_th)) * sin(HI_th / 2) /
           (2 * sqrt(pi)))

    hYb = (exp(1j * HI_ph / 2) * (-1 + 3 * cos(HI_th)) * cos(HI_th / 2) /
           (2 * sqrt(pi)))

    # transform to coefficient space

    # all coefficients other than those with l=3/2, m=\pm1/2 should be at
    # roundoff
    halm = ms.sf_to_salm(s=1, f=hYa - hYb, is_half_integer=True)

    # quick check by transforming back to nodal representation
    HI_N_th, HI_N_ph = ms.L_to_N(L_th=HI_L_th, L_ph=HI_L_ph,
                                 is_half_integer=True)
    h_rec = ms.salm_to_sf(s=1, alm=halm, N_th=HI_N_th, N_ph=HI_N_ph,
                          is_half_integer=True)

    print('@ max|h_rec - (hYa - hYb)) = {}'.format(
        max(abs(h_rec - (hYa - hYb)))))

    # extract specific entries using the index maps
    halm_nz_idx = ms.arr_idx_map(l=array([3, 3]), m=array([-1, 1]),
                                 is_half_integer=True)

    print('@ copying and removing +1, -1 non-zero coeffs')
    # copy & remove non-zero coefficients and check the max abs of remainder
    halm_c = halm.copy()
    halm_c[0, halm_nz_idx] = 0
    print('@ max|halm| \ ((3/2, -1/2), (3/2, 1/2)) = {}'.format(
        max(abs(halm_c), 1)))

    ###########################################################################
    # we apply the eth operators to the above functions in coefficient space
    #
    # Recall:
    #     \eth+: sYlm -> -sqrt((l - s)(l + s + 1)) {s+1}Ylm
    #     \eth-: sYlm ->  sqrt((l + s)(l - s + 1)) {s-1}Ylm
    #
    # So in coefficient space we can think of this as a simple mask
    ###########################################################################

    # (dense) representation of operation
    eth_arr = ms.eth_build(s=1, L=HI_L_th, type=-1, is_half_integer=True)
    n_eth_m_halm = eth_arr * halm  # s : 1/2 -> -1/2

    # sample analytical form of functions and multiply by corret front-factor:
    # ethm_hYa: s=-1/2, l=3/2, m=-1/2; -sqrt((3/2 - 1/2)(3 / 2 + 1/2 + 1))
    # ethm_hYb: s=-1/2, l=3/2, m=1/2;

    # fF = -sqrt((3 / 2 - 1 / 2) * (3 / 2 + 1 / 2 + 1))
    ethm_hYa = (exp(-1j * HI_ph / 2) * (-1 + 3 * cos(HI_th)) * cos(HI_th / 2) /
                sqrt(pi))

    ethm_hYb = -(exp(1j * HI_ph / 2) * (1 + 3 * cos(HI_th)) * sin(HI_th / 2) /
                 sqrt(pi))

    # transform and compare in coefficient space
    an_eth_m_halm = ms.sf_to_salm(s=-1, f=ethm_hYa - ethm_hYb,  # s now -1/2
                                  is_half_integer=True)

    print('@ max|n_eth_m_halm - an_eth_m_halm| = {}'.format(
        max(abs(n_eth_m_halm - an_eth_m_halm))))


def int_ex():
    '''
    Integer eth test; here we work as above but with far less detail.
    '''
    L_th, L_ph = 8, 8
    th, ph = ms.build_grid(L_th=L_th, L_ph=L_ph, is_half_integer=False)
    (th, ph) = (th[:, None], ph[None, :])
    # broadcast
    # Ya: s=2, l=4, m=-3
    # hYb: s=2, l=7, m=6
    Ya = (3 * exp(-3 * 1j * ph) * sqrt(7 / (2 * pi)) * cos(th / 2) *
          (1 + 2 * cos(th)) * sin(th / 2) ** 5)
    Yb = (exp(6 * 1j * ph) / 2 * sqrt(2145 / pi) * cos(th / 2) ** 8 *
          (-2 + 7 * cos(th)) * sin(th / 2) ** 4)

    alm = ms.sf_to_salm(s=2, f=Ya - Yb, is_half_integer=False)

    N_th, N_ph = ms.L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=False)
    rec = ms.salm_to_sf(s=2, alm=alm, N_th=N_th, N_ph=N_ph,
                        is_half_integer=False)

    print('@ max|rec - (Ya - Yb)) = {}'.format(max(abs(rec - (Ya - Yb)))))
    alm_nz_idx = ms.arr_idx_map(l=array([4, 7]), m=array([-3, 6]),
                                is_half_integer=False)

    print('@ copying and removing (4, -3), (7, 6) non-zero coeffs')
    alm_c = alm.copy()
    alm_c[0, alm_nz_idx] = 0
    print('@ max|alm| \ ((4, -3), (7, 6)) = {}'.format(max(abs(alm_c), 1)))

    eth_arr = ms.eth_build(s=2, L=L_th, type=1, is_half_integer=False)
    n_eth_p_alm = eth_arr * alm  # s : 2 -> 3

    ethp_Ya = (-3 * exp(-3 * 1j * ph) * sqrt(7 / (2 * pi)) *
               (3 + 4 * cos(th)) * sin(th / 2) ** 6)
    ethp_Yb = (5 / 2 * exp(6 * 1j * ph) * sqrt(2145 / pi) * cos(th / 2) ** 9 *
               (-3 + 7 * cos(th)) * sin(th / 2) ** 3)

    an_eth_p_alm = ms.sf_to_salm(s=3, f=ethp_Ya - ethp_Yb,  # s now 3
                                 is_half_integer=False)

    print('@ max|n_eth_p_alm - an_eth_p_alm| = {}'.format(
        max(abs(n_eth_p_alm - an_eth_p_alm))))

if __name__ == '__main__':
    h_int_ex()
    int_ex()

#
# :D
#
