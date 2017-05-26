#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 22:08:20

@author: Boris Daszuta
@function: Sanity check: forward - backward (see examples for more information)
"""
from numpy import (arange, abs, max)
import pytest as pt

import multi_X_SWSH as ms

ERR_CONST = 1e-12


@pt.mark.parametrize("L_th,L_ph", [
    (32, 32),
    (8, 32),
    (32, 8)
])
def test_integer_SWSH_pairs(L_th, L_ph):
    s = arange(-4, 4 + 1)
    N_th, N_ph = ms.L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=False)

    rand_salm = ms.generate_random_salm(s=s, L_th=L_th, L_ph=L_ph,
                                        is_half_integer=False)

    sf = ms.salm_to_sf(s=s, alm=rand_salm, N_th=N_th, N_ph=N_ph,
                       is_half_integer=False)

    salm = ms.sf_to_salm(s=s, f=sf, is_half_integer=False)

    err = max(abs(rand_salm - salm))
    assert err < ERR_CONST


@pt.mark.parametrize("L_th,L_ph", [
    (33, 33),
    (11, 33),
    (33, 11)
])
def test_half_integer_SWSH_pairs(L_th, L_ph):
    s = arange(-5, 5 + 1, 2)
    # L_th = L_ph = 33
    N_th, N_ph = ms.L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=True)

    rand_salm = ms.generate_random_salm(s=s, L_th=L_th, L_ph=L_ph,
                                        is_half_integer=True)

    sf = ms.salm_to_sf(s=s, alm=rand_salm, N_th=N_th, N_ph=N_ph,
                       is_half_integer=True)

    salm = ms.sf_to_salm(s=s, f=sf, is_half_integer=True)

    err = max(abs(rand_salm - salm))
    assert err < ERR_CONST

#
# :D
#
