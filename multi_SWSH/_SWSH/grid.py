#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience function for construction of a numerical grid based on a
supplied band-limit.
"""
import numpy as _np
import numba as _nu

from .idx_arr import L_to_N
from .._settings import _JIT_KWARGS


# @_nu.jit(**_JIT_KWARGS)
def build_grid(L_th=None, L_ph=None, is_half_integer=True, extended=False):
    '''
    Construct 'th', 'ph' one dimensional grids with optional extension from S2.
    '''
    N_th, N_ph = L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=is_half_integer)

    if is_half_integer:
        # extension in th,ph requires more points
        N_th_E, N_th_EE = 2 * (N_th - 1), 4 * (N_th - 1)
        N_ph_E = 2 * N_ph

        # extended domain lattice
        h_int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_EE - 1) + 1)
        h_int_ph = 2 * _np.pi / N_ph * _np.arange(0, N_ph_E)

        if extended:
            return h_int_th, h_int_ph

        return h_int_th[:N_th], h_int_ph[:N_ph]

    # extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # extended domain lattice
    int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_E))
    int_ph = 2 * _np.pi / N_ph * _np.arange(0, (N_ph - 1) + 1)

    if extended:
        return int_th, int_ph
    return int_th[:N_th], int_ph[:N_ph]

#
# :D
#
