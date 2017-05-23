#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience functions for populating coefficients with random entries
for testing.
"""
import numba as _nu
import numpy as _np

from .._types import _INT_PREC

@_nu.jit(nopython=True, nogil=True, cache=True)
def _int_strip_tr(num_fun_i, s_arr_i, L_r_th_i, L_r_ph_i, L_th_tr, salm_i):
    # Strip values that should be zero based on truncation
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


@_nu.jit(nopython=True, nogil=True, cache=True)
def _int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    # Generate test coefficients with (uniform-random) entries in:
    # re(salm)\in[-1,1] /\ im(salm)\in[-1,1]

    if L_th_tr is None:
        L_th_tr = L_r_th

    # >s_arr ~ Spin-weights
    # >L_r_th ~ Maximal L_th to populate
    # >L_r_ph ~ Maximal L_ph to populate
    # >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    num_fun = s_arr.shape[0]

    # begin by dense construction:
    sz_salm = (L_th_tr + 1)**2

    # salm = _np.zeros((num_fun, sz_salm), dtype=_COMPLEX_PREC)
    salm = (1 - 2 * _np.random.rand(num_fun, sz_salm)) + \
        1j * (1 - 2 * _np.random.rand(num_fun, sz_salm))

    _int_strip_tr(num_fun, s_arr, L_r_th, L_r_ph, L_th_tr, salm)
    return salm


@_nu.jit(nopython=True, nogil=True, cache=True)
def _h_int_strip_tr(num_fun_i, s_arr_i,
                    L2_r_th_i, L2_r_ph_i, L2_th_tr, h_salm_i):
    # Strip values that should be zero based on truncation
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
                ind_lm = _INT_PREC(
                    1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                h_salm_i[f, ind_lm] = 0

        # l>L_r_th should be 0
        for l2 in _np.arange(L2_r_th_i + 2, L2_th_tr + 1, 2):
            for m2 in _np.arange(-l2, l2 + 1, 2):
                # Index for salm
                ind_lm = _INT_PREC(
                    1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))
                h_salm_i[f, ind_lm] = 0


@_nu.jit(nopython=True, nogil=True, cache=True)
def _h_int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    # Generate test coefficients with (uniform-random) entries in:
    # re(salm)\in[-1,1] /\ im(salm)\in[-1,1]

    if L_th_tr is None:
        L_th_tr = L_r_th

    s_arr = s_arr / 2
    L_r_th = L_r_th / 2
    L_r_ph = L_r_ph / 2
    L_th_tr = L_th_tr / 2
    # >s_arr ~ Spin-weights
    # >L_r_th ~ Maximal L_th to populate
    # >L_r_ph ~ Maximal L_ph to populate
    # >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    num_fun = s_arr.shape[0]

    L2_r_th, L2_r_ph, L2_th_tr = 2 * L_r_th, 2 * L_r_ph, 2 * L_th_tr

    # begin by dense construction:
    h_sz_salm = _INT_PREC(3 / 4 + (L2_th_tr / 2) * (2 + L2_th_tr / 2))

    # salm = _np.zeros((num_fun, sz_salm), dtype=_COMPLEX_PREC)
    h_salm = ((1 - 2 * _np.random.rand(num_fun, h_sz_salm)) +
              1j * (1 - 2 * _np.random.rand(num_fun, h_sz_salm)))

    _h_int_strip_tr(num_fun, s_arr, L2_r_th, L2_r_ph, L2_th_tr, h_salm)
    return h_salm


#
# :D
#
