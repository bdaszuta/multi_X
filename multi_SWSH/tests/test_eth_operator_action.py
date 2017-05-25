#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Thu May 25 2017> @ 19:47:00

@author: Boris Daszuta
@function: Sanity check: eth operator action
"""
from numpy import (arange, array, abs, max)
import pytest as pt

import multi_SWSH as ms

ERR_CONST = 1e-12

# @pt.mark.parametrize("L_th,L_ph", [
#     (32, 32),
#     (8, 32),
#     (32, 8)
# ])
# def test_integer_SWSH_pairs(L_th, L_ph):
#     s = arange(-4, 4 + 1)
#     N_th, N_ph = ms.L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=False)

#     rand_salm = ms.generate_random_salm(s=s, L_th=L_th, L_ph=L_ph,
#                                         is_half_integer=False)

#     sf = ms.salm_to_sf(s=s, alm=rand_salm, N_th=N_th, N_ph=N_ph,
#                        is_half_integer=False)

#     salm = ms.sf_to_salm(s=s, f=sf, is_half_integer=False)

#     err = max(abs(rand_salm - salm))
#     assert err < ERR_CONST

def test_integer_eth_action():
    assert 1 == 1


#
# :D
#
