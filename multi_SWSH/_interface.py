#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:11:23

@author: Boris Daszuta
@function: Interface for making use of the SWSH transformations;
see read.nfo.
"""
import numpy as _np
import numba as _nu

from . import _SWSH


_COMPLEX_PREC = _np.complex128
_INT_PREC = _np.int64


def int_salm(s_arr, f_arr):
    '''
    Compute salm from integer spin-weighted functions sampled on S2
    '''
    # ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_INT_PREC)
    if len(f_arr.shape) == 2:
        f_arr = _np.array([f_arr], dtype=_COMPLEX_PREC)
    return _SWSH.SWSH_salm.int_salm(s_arr, f_arr)


def h_int_salm(s_arr, f_arr):
    '''
    Compute salm from half-integer spin-weighted functions sampled on S2
    '''

    # Ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_np.float64)
    if len(f_arr.shape) == 2:
        f_arr = _np.array([f_arr], dtype=_COMPLEX_PREC)

    return _SWSH.SWSH_salm.h_int_salm(s_arr, f_arr)


def int_sf(s_arr, salm_arr_i, N_th, N_ph):
    '''
    Compute integer spin-weighted functions on S2 from salm
    '''
    salm_arr = _np.copy(salm_arr_i)
    # Ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_np.float64)
    if len(salm_arr.shape) == 1:
        salm_arr = _np.array([salm_arr], dtype=_COMPLEX_PREC)

    return _SWSH.SWSH_bwd.int_sf(s_arr, salm_arr, N_th, N_ph)


def h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph):
    '''
    Compute integer spin-weighted functions on S2 from salm
    '''

    # Ensure we deal with arrays
    if _np.isscalar(h_s_arr):
        h_s_arr = _np.array([h_s_arr], dtype=_np.float64)
    if len(h_salm_arr.shape) == 1:
        h_salm_arr = _np.array([h_salm_arr], dtype=_COMPLEX_PREC)

    return _SWSH.SWSH_bwd.h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph)


def int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    '''
    Generate test coefficients with (uniform-random) entries in:
    re(salm)\in[-1,1] /\ im(salm)\in[-1,1]

    >s_arr ~ Spin-weights
    >L_r_th ~ Maximal L_th to populate
    >L_r_ph ~ Maximal L_ph to populate
    >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    '''
    num_fun = s_arr.shape[0]

    # begin by dense construction:
    sz_salm = (L_th_tr + 1)**2

    # salm = _np.zeros((num_fun, sz_salm), dtype=_COMPLEX_PREC)

    salm = (1 - 2 * _np.random.rand(num_fun, sz_salm)) + \
        1j * (1 - 2 * _np.random.rand(num_fun, sz_salm))

    @_nu.jit(nopython=True, nogil=True, cache=True)
    def _strip_tr(num_fun_i, s_arr_i, L_r_th_i, L_r_ph_i, L_th_tr, salm_i):
        '''
        Strip values that should be zero based on truncation
        '''
        for f in _np.arange(0, num_fun_i):
            # l<|s| should be 0
            for l in _np.arange(0, _np.abs(s_arr_i[f])):
                for m in _np.arange(-l, l + 1):
                    # Index for salm
                    ind_lm = l * (l + 1) + m
                    salm_i[f, ind_lm] = 0

            # m>L_r_ph should be 0
            for l in _np.arange(_np.abs(s_arr_i[f]), L_r_th_i + 1):
                for m in _np.arange(-l, -L_r_ph_i):
                    # Index for salm
                    ind_lm = l * (l + 1) + m
                    salm_i[f, ind_lm] = 0

                for m in _np.arange(L_r_ph_i + 1, l + 1):
                    # Index for salm
                    ind_lm = l * (l + 1) + m
                    salm_i[f, ind_lm] = 0

            # l>L_r_th should be 0
            for l in _np.arange(L_r_th_i + 1, L_th_tr + 1):
                for m in _np.arange(-l, l + 1):
                    # Index for salm
                    ind_lm = l * (l + 1) + m
                    salm_i[f, ind_lm] = 0

    _strip_tr(num_fun, s_arr, L_r_th, L_r_ph, L_th_tr, salm)

    return salm


def h_int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    '''
    Generate test coefficients with (uniform-random) entries in:
    re(salm)\in[-1,1] /\ im(salm)\in[-1,1]

    >s_arr ~ Spin-weights
    >L_r_th ~ Maximal L_th to populate
    >L_r_ph ~ Maximal L_ph to populate
    >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    '''
    num_fun = s_arr.shape[0]

    L2_r_th, L2_r_ph, L2_th_tr = 2 * L_r_th, 2 * L_r_ph, 2 * L_th_tr

    # begin by dense construction:
    h_sz_salm = _INT_PREC(3 / 4 + (L2_th_tr / 2) * (2 + L2_th_tr / 2))

    # salm = _np.zeros((num_fun, sz_salm), dtype=_COMPLEX_PREC)
    h_salm = ((1 - 2 * _np.random.rand(num_fun, h_sz_salm)) +
              1j * (1 - 2 * _np.random.rand(num_fun, h_sz_salm)))

    @_nu.jit(nopython=True, nogil=True, cache=True)
    def _strip_tr(num_fun_i, s_arr_i,
                  L2_r_th_i, L2_r_ph_i, L2_th_tr, h_salm_i):
        '''
        Strip values that should be zero based on truncation
        '''
        for f in _np.arange(0, num_fun_i):
            # l<|s| should be 0
            for l2 in _np.arange(1, _INT_PREC(_np.abs(2 * s_arr_i[f])), 2):
                for m2 in _np.arange(-l2, l2 + 1, 2):
                    # Index for salm
                    ind_lm = _INT_PREC(
                        1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                    h_salm_i[f, ind_lm] = 0

            # m>L_r_ph should be 0
            for l2 in _np.arange(_INT_PREC(_np.abs(2 * s_arr_i[f])),
                                 L2_r_th_i + 1, 2):
                for m2 in _np.arange(-l2, -L2_r_ph_i, 2):
                    # Index for salm
                    ind_lm = _INT_PREC(
                        1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                    h_salm_i[f, ind_lm] = 0

                for m2 in _np.arange(L2_r_ph_i + 2, l2 + 1, 2):
                    # Index for salm
                    ind_lm = _INT_PREC(1 / 2 *
                                       (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                    h_salm_i[f, ind_lm] = 0

            # l>L_r_th should be 0
            for l2 in _np.arange(L2_r_th_i + 2, L2_th_tr + 1, 2):
                for m2 in _np.arange(-l2, l2 + 1, 2):
                    # Index for salm
                    ind_lm = _INT_PREC(1 / 2 *
                                       (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                    h_salm_i[f, ind_lm] = 0

    _strip_tr(num_fun, s_arr, L2_r_th, L2_r_ph, L2_th_tr, h_salm)
    return h_salm


#
# :D
#
