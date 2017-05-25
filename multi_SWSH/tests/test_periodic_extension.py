#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Thu May 25 2017> @ 13:27:52

@author: Boris Daszuta
@function: Check function extension and grid building.
"""
from numpy import (abs, cos, exp, max, pi, sin, sqrt)

import multi_SWSH as ms

ERR_CONST = 1e-12


def test_integer_SWSH_extension():
    s = 2

    def int_fun(th, ph):
        return (3 * exp(-1j * ph) *
                (3 * sin(th) + 2 * sin(2 * th) +
                 7 * sin(3 * th) - 7 * sin(4 * th)) /
                (32 * sqrt(2 * pi)))

    # evaluate on grid [This requires an even number of samples]
    N_th, N_ph = 4, 6
    L_th, L_ph = ms.N_to_L(N_th=N_th, N_ph=N_ph, is_half_integer=False)
    th, ph = ms.build_grid(L_th=L_th, L_ph=L_ph, is_half_integer=False,
                           is_extended=True)

    # we sample the function on the extended domain directly
    int_fun_N = int_fun(th[:, None], ph[None, :])

    # restrict to fundamental (S2) domain
    int_fun_N_S2 = int_fun_N[0:N_th, :]

    # perform extension
    int_fun_N_ext = ms.sf_periodic_extension(s=s, sf=int_fun_N_S2,
                                             is_half_integer=False)

    # check difference in extension and full analytical specification on torus
    diff_int_ext = max(abs(int_fun_N_ext - int_fun_N))

    assert diff_int_ext < ERR_CONST


def test_half_integer_SWSH_extension():
    h_s = -3 / 2

    def h_int_fun(th, ph):
        return (1 / 40 * exp(-3 / 2 * 1j * ph) *
                sqrt(3 / pi) *
                cos(th / 2) * (
                    -20 * exp(2 * 1j * ph) *
                    (-1 + cos(th)) +
                    11 * sqrt(2) *
                    cos(th / 2)**2 * (-3 + 5 * cos(th))))

    # Evaluate on grid [here we put an even number of samples]
    # We will however pass an _odd_ number of samples to the extension fcn.
    h_N_th, h_N_ph = 4, 6

    L_th, L_ph = ms.N_to_L(N_th=h_N_th, N_ph=h_N_ph, is_half_integer=True)
    th, ph = ms.build_grid(L_th=L_th, L_ph=L_ph, is_half_integer=True,
                           is_extended=True)

    h_int_fun_N = h_int_fun(th[:, None], ph[None, :])

    # restrict to fundamental (S2) domain
    h_int_fun_N_S2 = h_int_fun_N[0:h_N_th, 0:h_N_ph]

    # perform extension
    h_int_fun_N_ext = ms.sf_periodic_extension(s=h_s, sf=h_int_fun_N_S2,
                                               is_half_integer=True)

    # check difference in extension and full analytical specification on torus
    diff_h_int_ext = max(abs(h_int_fun_N_ext - h_int_fun_N))

    assert diff_h_int_ext < ERR_CONST

#
# :D
#
