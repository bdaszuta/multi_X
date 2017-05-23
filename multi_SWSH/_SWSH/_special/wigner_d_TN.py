#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Thu Dec 17 21:40:18 2015

@author: Boris Daszuta
@function: Wigner-d recursion at half-pi for (half) integer elements.

TN-style algorithm.
"""
import numpy as _np
import numba as _nu

from ..._types import (_REAL_PREC, _INT_PREC)
from ..._settings import _JIT_KWARGS


# Init. values for recursion
_INIT_H_INT = 1 / _np.sqrt(_np.array([2], dtype=_REAL_PREC))
_INIT_INT = _np.array([1], dtype=_REAL_PREC)


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Strip(L2):
    '''
    Compute edge strip for half-integer elements up to a maximal 'L'

    We initialise with l=1/2, m=1/2, n=1/2 and generate all valid half-integer
    symbols Del^l_{l,1/2} up to Del^L_{L,1/2}

    >L2 ~ Twice the maximal 'L' to compute (where L is half-integer)
    '''
    L_num = _INT_PREC((L2 + 1) / 2)
    h_int_strip = _np.zeros(L_num, dtype=_REAL_PREC)

    # init. val
    h_int_strip[0] = _INIT_H_INT[0]

    for l_ind in _np.arange(1, L_num):
        l = _REAL_PREC((2 * l_ind + 1) / 2)

        h_int_strip[l_ind] = _np.sqrt(
            2 * l / (2 * l + 1)) * h_int_strip[l_ind - 1]

    return h_int_strip


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Exterior_full(L2, h_strip):
    '''
    Compute exterior octant for half-integer elements up to a maximal 'L'

    >L2 ~ Twice the maximal 'L' to compute (where L is half-integer)
    >h_strip ~ edge strip i_nput computed via h_int_Del_Strip
    '''
    L_num_ext = _INT_PREC((L2 + 1) * (L2 + 3) / 8)
    L_num_strip = _INT_PREC((L2 + 1) / 2)

    h_int_ext = _np.zeros(L_num_ext, dtype=_REAL_PREC)

    # init. val
    h_int_ext[0] = _INIT_H_INT[0]

    # populate strip values
    for l_ind in _np.arange(1, L_num_strip):
        # Calculate current index
        l, n = 2 * l_ind + 1, 1

        # lsq = l_sq_arr[_INT_PREC((l-1)/2)]
        ind_cur = _INT_PREC((l**2 + 4 * n - 5) / 8)

        h_int_ext[ind_cur] = h_strip[l_ind]

    # calculate edge values
    for l_ind in _np.arange(1, L_num_strip):
        l = 2 * l_ind + 1
        lsq = l**2
        l_o_sq = (l - 2)**2
        for n_ind in _np.arange(3, 2 * l_ind + 1 + 1, 2):

            # calculate current index
            n = n_ind

            ind_cur = _INT_PREC((lsq + 4 * n - 5) / 8)

            # old index
            l_o, n_o = l - 2, n - 2
            ind_o = _INT_PREC((l_o_sq + 4 * n_o - 5) / 8)

            h_int_ext[ind_cur] = (
                _np.sqrt((l / 2) * (l - 1) / ((l + n) * ((l + n) / 2 - 1))) *
                h_int_ext[ind_o])

    return h_int_ext


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Exterior_tr(L2, h_strip, N2_tr):
    '''
    Truncate del_mn on n index for a maximal N_tr
    (see h_int_Del_Exterior_full)
    '''
    L_num_ext = _INT_PREC((3 + 2 * L2 - N2_tr) * (N2_tr + 1) / 8)
    L_num_strip = _INT_PREC((L2 + 1) / 2)

    h_int_ext = _np.zeros(L_num_ext, dtype=_REAL_PREC)

    # init. val
    h_int_ext[0] = _INIT_H_INT[0]

    # populate strip values
    for l_ind in _np.arange(1, L_num_strip):
        # calculate current index
        l2, n2 = 2 * l_ind + 1, 1

        if l_ind <= _INT_PREC((N2_tr - 1) / 2):
            ind_cur = _INT_PREC((l2**2 + 4 * n2 - 5) / 8)
        else:
            ind_cur = _INT_PREC(
                (-5 + 4 * n2 + 2 * l2 * (N2_tr + 1) - N2_tr * (N2_tr + 2)) / 8)

        h_int_ext[ind_cur] = h_strip[l_ind]

    # calculate edge values (non-truncated)
    for l2 in _np.arange(1, N2_tr + 1, 2):
        # l = 2*l_ind+1
        lsq = l2**2
        l_o_sq = (l2 - 2)**2
        for n_ind in _np.arange(3, l2 + 1, 2):

            # calculate current index
            n2 = n_ind

            ind_cur = _INT_PREC((lsq + 4 * n2 - 5) / 8)

            # old index
            l_o, n_o = l2 - 2, n2 - 2

            ind_o = _INT_PREC((l_o_sq + 4 * n_o - 5) / 8)

            h_int_ext[ind_cur] = (
                _np.sqrt((l2 / 2) *
                         (l2 - 1) / ((l2 + n2) * ((l2 + n2) / 2 - 1))) *
                h_int_ext[ind_o])

    # intermediate values
    l2 = N2_tr + 2
    lsq = l2**2
    l_o_sq = (l2 - 2)**2
    for n2 in _np.arange(3, N2_tr + 1, 2):

        ind_cur = _INT_PREC(
            (-5 + 4 * n2 + 2 * l2 * (N2_tr + 1) - N2_tr * (N2_tr + 2)) / 8)

        # old index
        l_o, n_o = l2 - 2, n2 - 2

        ind_o = _INT_PREC((l_o_sq + 4 * n_o - 5) / 8)

        h_int_ext[ind_cur] = (
            _np.sqrt((l2 / 2) *
                     (l2 - 1) / ((l2 + n2) * ((l2 + n2) / 2 - 1))) *
            h_int_ext[ind_o])

    # calculate edge values (truncated)
    for l2 in _np.arange(N2_tr + 2, L2 + 1, 2):

        # l = 2*l_ind+1
        lsq = l2**2
        l_o_sq = (l2 - 2)**2
        for n2 in _np.arange(3, N2_tr + 1, 2):

            ind_cur = _INT_PREC(
                (-5 + 4 * n2 + 2 * l2 * (N2_tr + 1) - N2_tr * (N2_tr + 2)) / 8)

            # old index
            ind_o = _INT_PREC(
                (-5 + 4 * (n2 - 2) +
                 2 * (l2 - 2) * (N2_tr + 1) - N2_tr * (N2_tr + 2)) / 8)

            h_int_ext[ind_cur] = (
                _np.sqrt((l2 / 2) *
                         (l2 - 1) / ((l2 + n2) * ((l2 + n2) / 2 - 1))) *
                h_int_ext[ind_o])

    return h_int_ext


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Exterior(L2, h_int_strip, N2_tr=None):
    '''
    Handler for exterior (face) truncation
    '''
    if N2_tr is None:
        return h_int_Del_Exterior_full(L2, h_int_strip)
    elif N2_tr >= L2:
        return h_int_Del_Exterior_full(L2, h_int_strip)
    else:
        return h_int_Del_Exterior_tr(L2, h_int_strip, N2_tr)


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Interior_full(L2, h_ext):
    '''
    Compute interior octant for half-integer elements of fixed 'L'

    >L2 ~ Twice the desired 'L' to compute (where L is half-integer)
    >h_ext ~ exterior octant face computed via h_int_Del_Exterior
    '''

    L_num_int = _INT_PREC((L2 + 1) * (L2 + 3) / 8)
    L_num_strip = _INT_PREC((L2 + 1) / 2)

    h_int_int = _np.zeros(L_num_int, dtype=_REAL_PREC)

    # pre-calculate some re-used factors
    L2fac = L2 * (4 + L2)

    # generator dealt with separately
    if L_num_strip == 1:
        h_int_int[0] = h_ext[0]

    # populate the edge values
    for l_ind in _np.arange(1, L_num_strip):
        l = 2 * l_ind + 1
        lsq = l**2
        for n_ind in _np.arange(1, 2 * l_ind + 1 + 1, 2):

            # calculate current index [external indexing]
            n = n_ind
            ind_cur = _INT_PREC((lsq + 4 * n - 5) / 8)

            # calculate current index [internal indexing]
            m = l
            ind_cur_int = _INT_PREC((4 * l + lsq - m**2 - 4 * n) / 8)

            h_int_int[ind_cur_int] = h_ext[ind_cur]

    for n in _np.arange(1, L2 + 1, 2):
        for m in _np.arange(L2 - 2, n - 1, -2):
            msq = m**2
            mp2sq = msq + 4 * m + 4
            mp4sq = msq + 8 * m + 16

            if m == L2 - 2:
                # current index
                ind_cur = _INT_PREC((L2fac - msq - 4 * n) / 8)
                ind_mp2 = _INT_PREC((L2fac - mp2sq - 4 * n) / 8)

                h_int_int[ind_cur] = - \
                    (n / 2) * _np.sqrt(4 / L2) * h_int_int[ind_mp2]

            else:
                # current index
                ind_cur = _INT_PREC((L2fac - msq - 4 * n) / 8)
                ind_mp2 = _INT_PREC((L2fac - mp2sq - 4 * n) / 8)
                ind_mp4 = _INT_PREC((L2fac - mp4sq - 4 * n) / 8)

                h_int_int[ind_cur] = (
                    -n / _np.sqrt((l - m) / 2 * ((l + m) / 2 + 1)) *
                    h_int_int[ind_mp2] -
                    _np.sqrt(((l - m) / 2 - 1) * ((l + m) / 2 + 2) /
                             ((l - m) / 2 * ((l + m) / 2 + 1))) *
                    h_int_int[ind_mp4])

    return h_int_int


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Interior_tr(L2, h_ext_tr, N2_tr):
    '''
    Compute interior octant for half-integer elements of fixed 'L'

    >L2 ~ Twice the desired 'L' to compute (where L is half-integer)
    >h_ext ~ exterior octant face computed via h_int_Del_Exterior

    Note: it is assumed that L2>N2_tr
    '''
    # we assume that L2>N2_tr else just call the specific routine without
    # truncation.
    L_num_int = _INT_PREC((L2 + 1) * (N2_tr + 1) / 4)
    h_int_int = _np.zeros(L_num_int, dtype=_REAL_PREC)

    # populate the edge values
    for n2 in _np.arange(1, N2_tr + 1, 2):
        # calculate current index [external indexing]
        ind_cur = _INT_PREC(
            (-5 + 4 * n2 + 2 * L2 * (N2_tr + 1) - N2_tr * (N2_tr + 2)) / 8)

        # calculate current index [internal indexing]
        ind_cur_int = _INT_PREC((L2 + 1) * (n2 - 1) / 4 + (L2 - 1) / 2)

        h_int_int[ind_cur_int] = h_ext_tr[ind_cur]

    # compute internal values
    for n2 in _np.arange(1, N2_tr + 1, 2):

        L2_fac = (L2 + 1) * (n2 - 1) / 4
        for m2 in _np.arange(L2 - 2, -1, -2):
            if m2 == L2 - 2:
                # current index
                ind_cur = _INT_PREC(L2_fac + (m2 - 1) / 2)
                ind_mp2 = _INT_PREC(L2_fac + ((m2 + 2) - 1) / 2)

                h_int_int[ind_cur] = -(n2 / 2) * \
                    _np.sqrt(4 / L2) * h_int_int[ind_mp2]

            else:
                # current index
                ind_cur = _INT_PREC(L2_fac + (m2 - 1) / 2)
                ind_mp2 = _INT_PREC(L2_fac + ((m2 + 2) - 1) / 2)
                ind_mp4 = _INT_PREC(L2_fac + ((m2 + 4) - 1) / 2)

                h_int_int[ind_cur] = (
                    -n2 / _np.sqrt((L2 - m2) / 2 * ((L2 + m2) / 2 + 1)) *
                    h_int_int[ind_mp2] -
                    _np.sqrt(((L2 - m2) / 2 - 1) * ((L2 + m2) / 2 + 2) /
                             ((L2 - m2) / 2 * ((L2 + m2) / 2 + 1))) *
                    h_int_int[ind_mp4])

    return h_int_int


@_nu.jit(**_JIT_KWARGS)
def h_int_Del_Interior(L2, h_int_ext, N2_tr=None):
    '''
    Handler for exterior (face) truncation
    '''
    if N2_tr is None:
        return h_int_Del_Interior_full(L2, h_int_ext)
    elif N2_tr >= L2:
        return h_int_Del_Interior_full(L2, h_int_ext)
    else:
        return h_int_Del_Interior_tr(L2, h_int_ext, N2_tr)


@_nu.jit(**_JIT_KWARGS)
def int_Del_Strip(L):
    '''
    Compute edge strip for integer elements up to a maximal 'L'

    We initialise with l=0, m=0, n=0

    >L ~ Maximal 'L' to compute
    '''
    L_num = _INT_PREC(L + 1)
    int_strip = _np.zeros(L_num, dtype=_REAL_PREC)

    # init. val
    int_strip[0] = _INIT_INT[0]

    for l_ind in _np.arange(1, L_num):
        l = _REAL_PREC(l_ind)
        int_strip[l_ind] = _np.sqrt(
            (2 * l - 1) / (2 * l)) * int_strip[l_ind - 1]

    return int_strip


@_nu.jit(**_JIT_KWARGS)
def int_Del_Exterior_full(L, int_strip):
    '''
    Compute exterior octant for integer elements up to a maximal 'L'

    >L ~ Maximal 'L' to compute
    >int_strip ~ edge strip i_nput computed via int_Del_Strip
    '''
    L_num_ext = _INT_PREC((L + 1) * (L + 2) / 2)
    L_num_strip = _INT_PREC(L + 1)

    int_ext = _np.zeros(L_num_ext, dtype=_REAL_PREC)

    # init. val
    int_ext[0] = _INIT_INT[0]

    # populate strip values
    for l_ind in _np.arange(1, L_num_strip):
        # calculate current index
        ind_cur = _INT_PREC((l_ind * (l_ind + 1)) / 2)
        int_ext[ind_cur] = int_strip[l_ind]

    # calculate edge values
    for l_ind in _np.arange(1, L_num_strip):
        for n_ind in _np.arange(1, l_ind + 1):
            ind_cur = _INT_PREC((l_ind * (l_ind + 1) + 2 * n_ind) / 2)
            ind_old = _INT_PREC(
                ((l_ind - 1) * (l_ind - 1 + 1) + 2 * (n_ind - 1)) / 2)

            fac = _np.sqrt(l_ind * (2 * l_ind - 1) /
                           (2 * (l_ind + n_ind) * (l_ind + n_ind - 1)))

            int_ext[ind_cur] = fac * int_ext[ind_old]

    return int_ext


@_nu.jit(**_JIT_KWARGS)
def int_Del_Exterior_tr(L, int_strip, N_tr):
    '''
    Truncate del_mn on n index for a maximal N_tr (see int_Del_Exterior_full)
    '''
    L_num_ext = _INT_PREC((2 + 2 * L - N_tr) * (N_tr + 1) / 2)
    L_num_strip = _INT_PREC(L + 1)

    int_ext = _np.zeros(L_num_ext, dtype=_REAL_PREC)

    # init. val
    int_ext[0] = _INIT_INT[0]

    # populate strip values
    for l_ind in _np.arange(1, L_num_strip):
        # calculate current index
        if l_ind > N_tr:
            ind_cur = _INT_PREC(l_ind * (N_tr + 1) - N_tr * (N_tr + 1) / 2)
        else:
            ind_cur = _INT_PREC((l_ind * (l_ind + 1)) / 2)

        int_ext[ind_cur] = int_strip[l_ind]

    # calculate edge values (non-truncated)
    for l_ind in _np.arange(1, N_tr + 1):

        for n_ind in _np.arange(1, l_ind + 1):
            ind_cur = _INT_PREC((l_ind * (l_ind + 1) + 2 * n_ind) / 2)
            ind_old = _INT_PREC(
                ((l_ind - 1) * (l_ind - 1 + 1) + 2 * (n_ind - 1)) / 2)

            fac = _np.sqrt(l_ind * (2 * l_ind - 1) /
                           (2 * (l_ind + n_ind) * (l_ind + n_ind - 1)))

            int_ext[ind_cur] = fac * int_ext[ind_old]

    # intermediate values
    l_ind = N_tr + 1
    for n_ind in _np.arange(1, N_tr + 1):
        ind_cur = _INT_PREC(n_ind + l_ind * (1 + N_tr) - N_tr * (N_tr + 1) / 2)
        ind_old = _INT_PREC(
            ((l_ind - 1) * (l_ind - 1 + 1) + 2 * (n_ind - 1)) / 2)

        fac = _np.sqrt(l_ind * (2 * l_ind - 1) /
                       (2 * (l_ind + n_ind) * (l_ind + n_ind - 1)))
        int_ext[ind_cur] = fac * int_ext[ind_old]

    for l_ind in _np.arange(N_tr + 2, L_num_strip):
        for n_ind in _np.arange(1, N_tr + 1):
            ind_cur = _INT_PREC(n_ind + l_ind * (1 + N_tr) -
                                N_tr * (N_tr + 1) / 2)
            ind_old = _INT_PREC((n_ind - 1) + (l_ind - 1) *
                                (1 + N_tr) - N_tr * (N_tr + 1) / 2)

            fac = _np.sqrt(l_ind * (2 * l_ind - 1) /
                           (2 * (l_ind + n_ind) * (l_ind + n_ind - 1)))

            int_ext[ind_cur] = fac * int_ext[ind_old]

    return int_ext


@_nu.jit(**_JIT_KWARGS)
def int_Del_Exterior(L, int_strip, N_tr=-1):
    '''
    Handler for exterior (face) truncation
    '''
    if N_tr == -1:
        return int_Del_Exterior_full(L, int_strip)
    elif N_tr >= L:
        return int_Del_Exterior_full(L, int_strip)

    return int_Del_Exterior_tr(L, int_strip, N_tr)


@_nu.jit(**_JIT_KWARGS)
def int_Del_Interior_full(L, int_ext):
    '''
    Compute interior octant for integer elements of fixed 'L'

    >L ~ current L slab to compute
    >int_ext ~ exterior octant face computed via int_Del_Exterior
    '''
    L_num_int = _INT_PREC(((L + 2) * (L + 1)) / 2)

    int_int = _np.zeros(L_num_int, dtype=_REAL_PREC)

    # populate the edge values
    for n_ind in _np.arange(0, L + 1):
        ind_cur = _INT_PREC(L - n_ind)
        ind_ext = _INT_PREC((L * (L + 1) + 2 * n_ind) / 2)

        int_int[ind_cur] = int_ext[ind_ext]

    # pull out factors
    Lf = L * (3 + L)

    # compute internal values
    for n in _np.arange(0, L + 1):
        for m in _np.arange(L - 1, n - 1, -1):
            if m == (L - 1):
                ind_cur = _INT_PREC((Lf - m * (m + 1) - 2 * n) / 2)
                ind_mp1 = _INT_PREC((Lf - (m + 1) * (m + 2) - 2 * n) / 2)
                int_int[ind_cur] = -_np.sqrt(2 / L) * n * int_int[ind_mp1]
            else:
                ind_cur = _INT_PREC((Lf - m * (m + 1) - 2 * n) / 2)
                ind_mp1 = _INT_PREC((Lf - (m + 1) * (m + 2) - 2 * n) / 2)
                ind_mp2 = _INT_PREC((Lf - (m + 2) * (m + 3) - 2 * n) / 2)

                int_int[ind_cur] = (
                    -2 * n / _np.sqrt((L - m) * (L + m + 1)) *
                    int_int[ind_mp1] -
                    _np.sqrt((L - m - 1) * (L + m + 2) /
                             ((L - m) * (L + m + 1))) * int_int[ind_mp2])

    return int_int


@_nu.jit(**_JIT_KWARGS)
def int_Del_Interior_tr(L, int_ext_tr, N_tr):
    '''
    Compute interior octant for integer elements of fixed 'L'

    >L ~ current L slab to compute
    >int_ext ~ exterior octant face computed via int_Del_Exterior

    Note: it is assumed that L>N_tr
    '''
    # we assume that L>N_tr else just call the specific routine without
    # truncation.
    L_num_int = _INT_PREC((L + 1) * (N_tr + 1))
    int_int = _np.zeros(L_num_int, dtype=_REAL_PREC)

    # populate the edge values
    for n_ind in _np.arange(0, N_tr + 1):
        ind_cur = _INT_PREC(L * (N_tr + 1) + n_ind)
        ind_ext = _INT_PREC(n_ind + L * (N_tr + 1) - N_tr * (N_tr + 1) / 2)

        int_int[ind_cur] = int_ext_tr[ind_ext]

    # compute internal values
    for n in _np.arange(0, N_tr + 1):
        # work towards truncation index
        for m in _np.arange(L - 1, -1, -1):
            if m == (L - 1):
                # nearest-neighbour to edge of lattice - single term
                ind_cur = _INT_PREC(m * (N_tr + 1) + n)
                ind_mp1 = _INT_PREC((m + 1) * (N_tr + 1) + n)

                int_int[ind_cur] = -_np.sqrt(2 / L) * n * int_int[ind_mp1]

            else:
                # interior of lattice - three-term recursion
                ind_cur = _INT_PREC(m * (N_tr + 1) + n)
                ind_mp1 = _INT_PREC((m + 1) * (N_tr + 1) + n)
                ind_mp2 = _INT_PREC((m + 2) * (N_tr + 1) + n)

                int_int[ind_cur] = (
                    -2 * n / _np.sqrt((L - m) * (L + m + 1)) *
                    int_int[ind_mp1] -
                    _np.sqrt((L - m - 1) * (L + m + 2) /
                             ((L - m) * (L + m + 1))) * int_int[ind_mp2])

    return int_int


@_nu.jit(**_JIT_KWARGS)
def int_Del_Interior(L, int_ext, N_tr=-1):
    '''
    Handler for exterior (face) truncation
    '''
    if N_tr == -1:
        return int_Del_Interior_full(L, int_ext)
    elif N_tr >= L:
        return int_Del_Interior_full(L, int_ext)
    else:
        return int_Del_Interior_tr(L, int_ext, N_tr)


@_nu.jit(**_JIT_KWARGS)
def _int_Del_Interior_ExtendQuad(l, cur_oct):
    '''
    Extend octant to quadrant using symmetries
    '''
    l = _INT_PREC(l)
    int_int_quad = _np.zeros((l + 1, l + 1), dtype=_REAL_PREC)

    # extract internal values
    l_fac = l * (3 + l)

    # octant and edge
    for n in _np.arange(0, l + 1):
        for m in _np.arange(l - 1, n - 1, -1):
            ind_cur = _INT_PREC((l_fac - m * (m + 1) - 2 * n) / 2)
            int_int_quad[m, n] = cur_oct[ind_cur]

        int_int_quad[l, n] = cur_oct[_INT_PREC(l - n)]

    # use symmetries to unfold to quadrant
    for n in _np.arange(0, l + 1):
        for m in _np.arange(0, n):
            # int_int_quad[m, n] = (-1)**(m-n)*int_int_quad[n, m]
            int_int_quad[m, n] = ((1 - 2 * _np.mod(m - n, 2)) *
                                  int_int_quad[n, m])
    return int_int_quad


@_nu.jit(**_JIT_KWARGS)
def int_Del_Interior_ExtendQuad(l, cur_oct, N_ph=-1):
    '''
    Interface that enables truncation.
    '''
    l = _INT_PREC(l)
    if N_ph == -1:
        return _int_Del_Interior_ExtendQuad(l, cur_oct)
    elif N_ph >= l:
        return _int_Del_Interior_ExtendQuad(l, cur_oct)

    del_ret = _np.zeros((l + 1, N_ph + 1), dtype=_REAL_PREC)

    for i in _np.arange(0, l + 1):
        for j in _np.arange(0, N_ph + 1):
            del_ret[i, j] = cur_oct[i * (N_ph + 1) + j]
    # return _np.reshape(cur_oct, (l+1, N_ph+1))
    return del_ret


@_nu.jit(**_JIT_KWARGS)
def int_Del_Interior_ExtendFull(l, cur_oct):
    '''
    Extend octant to full slab using symmetries
    '''
    int_int_full = _np.zeros((2 * l + 1, 2 * l + 1), dtype=_REAL_PREC)

    # Extract internal values
    l_fac = l * (3 + l)

    # octant and edge
    for n in _np.arange(0, l + 1):
        for m in _np.arange(l - 1, n - 1, -1):
            ind_cur = _INT_PREC((l_fac - m * (m + 1) - 2 * n) / 2)
            int_int_full[m + l, n + l] = cur_oct[ind_cur]

        int_int_full[_INT_PREC(2 * l), _INT_PREC(l + n)
                     ] = cur_oct[_INT_PREC(l - n)]

    # use symmetries to unfold to quadrant
    for n in _np.arange(0, l + 1):
        for m in _np.arange(0, n):
            int_int_full[l + m, l +
                         n] = (-1)**(m - n) * int_int_full[l + n, l + m]

    # use symmetries to unfold half
    for m in _np.arange(-l, 0):
        for n in _np.arange(0, l + 1):
            int_int_full[l + m, l +
                         n] = (-1)**(l + n) * int_int_full[l - m, l + n]

    # use symmetries to unfold full
    for m in _np.arange(-l, l + 1):
        for n in _np.arange(-l, 0):
            int_int_full[m + l, n +
                         l] = (-1)**(l - m) * int_int_full[m + l, l - n]

    return int_int_full


@_nu.jit(**_JIT_KWARGS)
def _h_int_Del_Interior_ExtendQuad(l2, cur_oct):
    '''
    Extend octant to quadrant using symmetries
    '''
    dim = _INT_PREC(_np.sqrt(1 / 4 + l2 / 2 * (l2 / 2 + 1)))
    h_int_int_quad = _np.zeros((dim, dim), dtype=_REAL_PREC)

    # use symmetries to populate interior
    for n in _np.arange(1, l2 + 2, 2):
        for m in _np.arange(1, l2 + 2, 2):

            if n == m:
                # calculate 1d index for octant vals.
                ind_cur = _INT_PREC((l2 * (4 + l2) - m**2 - 4 * n) / 8)

                # no symmetry required as we have a diag. val.
                m_i, n_i = _INT_PREC((m - 1) / 2), _INT_PREC((n - 1) / 2)

                h_int_int_quad[m_i, n_i] = cur_oct[ind_cur]
            elif n < m:
                # calculate 1d index for octant vals.
                ind_cur = _INT_PREC((l2 * (4 + l2) - m**2 - 4 * n) / 8)

                # no symmetry required as we have a diag. val.
                m_i, n_i = _INT_PREC((m - 1) / 2), _INT_PREC((n - 1) / 2)

                # insert value
                h_int_int_quad[m_i, n_i] = cur_oct[ind_cur]

                # apply symmetry
                h_int_int_quad[
                    n_i, m_i] = (-1)**((n - m) / 2) * cur_oct[ind_cur]

    return h_int_int_quad


def h_int_Del_Interior_ExtendQuad(l2, cur_oct, N2_ph=-1):
    '''
    Interface that enables truncation.
    '''
    if N2_ph == -1:
        return _h_int_Del_Interior_ExtendQuad(l2, cur_oct)
    elif N2_ph >= l2:
        return _h_int_Del_Interior_ExtendQuad(l2, cur_oct)
    else:
        return _np.reshape(
            cur_oct,
            (_INT_PREC((l2 + 1) / 2), _INT_PREC((N2_ph + 1) / 2)),
            order='F')


def _main():
    L = 16
    N_tr = 2
    int_Del_strip = int_Del_Strip(L)
    int_Del_ext = int_Del_Exterior(L, int_Del_strip)

    int_Del_ext_tr = int_Del_Exterior(L, int_Del_strip, N_tr)

    int_Del_int_tr = int_Del_Interior(6, int_Del_ext_tr, N_tr)
    for l in _np.arange(3, L + 1, 1):
        int_Del_int = int_Del_Interior(l, int_Del_ext)

    int_Del_ext_quad = int_Del_Interior_ExtendQuad(L + 1, int_Del_int)

    L2 = 13
    h_N2_tr = 7

    h_int_Del_strip = h_int_Del_Strip(L2)
    h_int_Del_ext = h_int_Del_Exterior(L2, h_int_Del_strip)

    h_int_Del_ext_tr = h_int_Del_Exterior(11, h_int_Del_strip, h_N2_tr)

    h_int_Del_int_tr = h_int_Del_Interior(11, h_int_Del_ext_tr, h_N2_tr)

    for l_c in _np.arange(3, L2 + 2, 2):
        h_int_Del_int = h_int_Del_Interior(l_c, h_int_Del_ext)

    h_int_Del_int = h_int_Del_Interior(L2, h_int_Del_ext)

    h_quad = h_int_Del_Interior_ExtendQuad(L2, h_int_Del_int)

if __name__ == '__main__':
    _main()


#
# :D
#
