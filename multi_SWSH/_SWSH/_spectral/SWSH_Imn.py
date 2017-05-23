#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Sat Dec 19 15:36:55 2015

@author: Boris Daszuta
@function: Quadrature for spin-weighted functions.
"""
import numpy as _np

from . import SWSH_periodic_extension
from ..._types import (_COMPLEX_PREC, _INT_PREC)


# Fourier object instance
FSi = None


def c_int_wei(p):
    '''
    Calculate the values of the weight function in theta direction.

    w(p) = \int^\pi_0 exp(ip\theta) sin\theta d\theta

    [Modal representation]
    '''
    # Alloc
    w_theta = _np.zeros_like(p, dtype=_COMPLEX_PREC)
    # p even
    w_theta[_np.mod(p, 2) == 0] = 2 / (1 - p[_np.mod(p, 2) == 0]**2)
    # p == \pm 1
    w_theta[_np.abs(p) == 1] = _np.sign(p[_np.abs(p) == 1]) * 1j * _np.pi / 2
    return w_theta


def int_wei_th(N_th_E):
    '''
    Transform values of weight function for I_mn

    [Nodal representation]
    '''
    p = _np.arange(-N_th_E / 2, (N_th_E / 2 - 1) + 1)
    wei = FSi.fftshift(c_int_wei(p))
    # Embed phi factor (2pi)
    n_wei = 2 * _np.pi * FSi.ifft(wei, normalized=True)
    n_wei = FSi.ifftshift(n_wei)

    return n_wei


def int_quadrature(s, int_sfun_N, n_wei):
    '''
    Calculate I_mn array
    '''
    # Extend function
    int_sfun_N_ext = SWSH_periodic_extension.int_funcExtension(s, int_sfun_N)

    N_ph = int_sfun_N.shape[1]

    # Create weight mask:
    n_wei_ma = _np.repeat(_np.transpose(_np.array([n_wei])), N_ph, axis=1)
    # Mask function with weight
    int_fun_N_mask_n_w = n_wei_ma * int_sfun_N_ext

    # Construct Imn [normalized]:
    I_mn = FSi.fft(int_fun_N_mask_n_w, normalized=True)

    # Shift DC freq to centre
    I_mn = FSi.fftshift(I_mn)

    return I_mn


def c_h_int_wei(t, p):
    '''
    Calculate the values of the weight function in theta and phi directions.

    [Modal representation]
    '''
    # Alloc
    w_theta = _np.zeros_like(t, dtype=_COMPLEX_PREC)
    w_phi = _np.zeros_like(p, dtype=_COMPLEX_PREC)

    abs_t = _np.abs(t)

    w_theta[abs_t == 2] = 1j * _np.pi / 2 * _np.sign(t[abs_t == 2])
    w_theta[abs_t != 2] = 4 * (1 + 1j**t[abs_t != 2]) / (4 - t[abs_t != 2]**2)

    w_phi[p == 0] = 2 * _np.pi
    w_phi[p != 0] = 2j / p[p != 0] * (1 - (-1)**p[p != 0])

    return w_theta, w_phi


def h_int_wei(N_th_EE, N_ph_E):
    t = _np.arange(-N_th_EE / 2, (N_th_EE / 2 - 1) + 1)
    p = _np.arange(-N_ph_E / 2, (N_ph_E / 2 - 1) + 1)

    wei_th, wei_ph = c_h_int_wei(t, p)

    # Shift freq. and transform
    wei_th = FSi.ifft(FSi.fftshift(wei_th), normalized=True)
    wei_th = _np.roll(wei_th, -1 + _INT_PREC((4 + N_th_EE) / 4), axis=0)

    wei_ph = FSi.ifft(FSi.fftshift(wei_ph), normalized=True)
    wei_ph = _np.roll(wei_ph, _INT_PREC(N_ph_E / 4), axis=0)
    return wei_th, wei_ph


def h_int_quadrature(s, h_int_sfun_N, n_wei_th, n_wei_ph):
    '''
    Calculate I_mn array
    '''
    # Extend function
    h_int_sfun_N_ext = SWSH_periodic_extension.h_int_funcExtension(
        s, h_int_sfun_N)

    N_th_EE, N_ph_E = h_int_sfun_N_ext.shape

    # Create weight mask:
    n_wei_ma = _np.repeat(_np.transpose(_np.array([n_wei_th])), N_ph_E, axis=1)
    n_wei_ma = n_wei_ma * _np.repeat(_np.array([n_wei_ph]), N_th_EE, axis=0)

    # Construct Imn [normalized]:
    h_I_mn = FSi.fft(n_wei_ma * h_int_sfun_N_ext, normalized=True)

    # Shift DC freq to centre
    h_I_mn = FSi.fftshift(h_I_mn)[1::2, 1::2]
    h_I_mn = h_I_mn[1:-1, :]
    return h_I_mn


def _main():
    # Remove this later
    import multi_SWSH._spectral.interface_Fourier_series as FS
    import multi_SWSH._special.wigner_d_AnSum as swsh_an

    ########
    # Integer
    ########

    # Num. nodes on S^2 grid [must be even]
    N_th, N_ph = 4, 4

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Extended domain lattice
    int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_E))
    int_ph = 2 * _np.pi / N_ph * _np.arange(0, (N_ph - 1) + 1)

    # Spin-weight
    s = 1

    # Test function [build from analytical defn. of Wigner D]
    int_fun = lambda th, ph: swsh_an.sylm(s, 2, 1, th, ph) + \
        11 / 10 * swsh_an.sylm(s, 3, 2, th, ph)

    # Evaluate function on 2-sphere
    int_fun_N_S2 = int_fun(int_th[:N_th], int_ph[:N_ph])

    # Computation of weights
    n_wei = int_wei_th(N_th_E)

    # Calculate I_mn
    I_mn = int_quadrature(s, int_fun_N_S2, n_wei)

    # Throw small values
    I_mn[_np.abs(I_mn) < 1e-12] = 0

    # Integer (m,n) tuple to shifted index fcn
    ind_tup = lambda m, n: (m + N_th_E / 2, n + N_ph / 2)

    # Provide values of m,n directly
    I_mn[ind_tup(-7, -1)]

    #############
    # Half-integer
    #############

    # Spin-weight
    h_s = 1 / 2

    # Test function [build from analytical defn. of Wigner D]
    h_int_fun = lambda th, ph: swsh_an.sylm(h_s, 1 / 2, -1 / 2, th, ph) + \
        11 / 10 * swsh_an.sylm(h_s, 3 / 2, 1 / 2, th, ph)

    # Num. nodes on S^2 grid [must be even]
    h_N_th, h_N_ph = 6, 6

    # Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2 * (h_N_th - 1), 4 * (h_N_th - 1)
    h_N_ph_E = 2 * h_N_ph

    # Extended domain lattice
    h_int_th = _np.pi / (h_N_th - 1) * _np.arange(0, (h_N_th_EE - 1) + 1)
    h_int_ph = 2 * _np.pi / h_N_ph * _np.arange(0, h_N_ph_E)

    # Evaluate function on 2-sphere
    h_int_fun_N = h_int_fun(h_int_th[:h_N_th], h_int_ph[:h_N_ph])
    h_int_sfun_N_ext = SWSH_periodic_extension.h_int_funcExtension(
        h_s, h_int_fun_N)

    # Computation of weights
    h_n_wei_th, h_n_wei_ph = h_int_wei(h_N_th_EE, h_N_ph_E)

    # Calculate I_mn
    h_I_mn = h_int_quadrature(h_s, h_int_fun_N, h_n_wei_th, h_n_wei_ph)

    # Throw small values
    h_I_mn[_np.abs(h_I_mn) < 1e-12] = 0

    # h = FSi.fftshift(h_I_mn)[1::2,1::2]
    # h = h[1:-1,:]
    # h = FSi.fftshift(h_I_mn)[5::2,1::2]

if __name__ == '__main__':
    _main()

#
# :D
#
