#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Fri May 26 2017> @ 03:19:56

@author: Boris Daszuta
@function: Test band-limit inference and array idx calculations
"""
import pytest as pt

import multi_X_SWSH as ms

ERR_CONST = 1e-12


@pt.mark.parametrize("L_th,L_ph,is_extended", [
    (5, 3, False),
    (11, 7, False),
    (10, 6, False),
    (11, 11, False),
    (12, 8, False),
    (32, 32, False)
])
def test_L_compatible(L_th, L_ph, is_extended):
    h_L_th, h_L_ph = ms.L_compatible(L_th=L_th, L_ph=L_ph,
                                     is_half_integer=False)

    gr = ms.build_grid(L_th=L_th, L_ph=L_ph, is_half_integer=False,
                       is_extended=is_extended)
    h_gr = ms.build_grid(L_th=h_L_th, L_ph=h_L_ph,
                         is_extended=is_extended)

    assert max(gr[0] - h_gr[0]) < ERR_CONST
    assert max(gr[1] - h_gr[1]) < ERR_CONST

    # now check opposite direction
    iL_th, iL_ph = ms.L_compatible(L_th=h_L_th, L_ph=h_L_ph,
                                   is_half_integer=True)

    assert iL_th == L_th
    assert iL_ph == L_ph

#
# :D
#
