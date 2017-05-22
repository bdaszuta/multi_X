#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Sat Dec 19 15:36:55 2015

@author: Boris Daszuta
@function: Periodic extension mechanism for spin-weighted functions.
"""
import numpy as _np
_INT_PREC = _np.int64


def int_funcExtension(s, sw_fun):
    '''
    Extend an integer spin-weight function on the 2-sphere to the 2-torus

    [0,pi]x[0,2pi] -> [0,2pi]x[0,2pi]

    Even number of samples required for extension.
    '''
    N_ph = sw_fun.shape[1]
    return _np.vstack(
        (sw_fun, (-1)**s * _np.flipud(_np.roll(sw_fun[1:-1, :],
                                               _np.int(N_ph / 2), 1))))


def int_funcDoubleExtension(s, sw_fun):
    '''
    Extend an integer spin-weight function on the 2-sphere to a doubly periodic
    function on the 2-torus.
    This may be useful during multiplication by half-integer spin-weight
    quantities.

    Even number of samples required for extension
    '''
    sw_fun_T2 = int_funcExtension(s, sw_fun)

    # now extend again:
    sw_fun_T2_E = _np.repeat(sw_fun_T2, 2, axis=1)
    return _np.repeat(sw_fun_T2_E, 2, axis=0)


def h_int_funcExtension(s, sw_fun):
    '''
    Extend a half-integer spin-weight function on the 2-sphere to the 2-torus

    [0,pi]x[0,2pi] -> [0,4pi]x[0,4pi]

    Even number of samples required for extension.
    '''

    N_th, N_ph = sw_fun.shape

    # extend to D_I U D_II
    sw_fun_E = _np.hstack((sw_fun, -sw_fun))

    s_int = _INT_PREC(2 * s)
    s_ph = -(1j**s_int)
    # s_ph = -(-1)**s
    # print([s_ph, -(-1)**s])

    # extend to D_III U D_IV
    sw_fun_E = _np.vstack(
        (sw_fun_E, s_ph * _np.roll(_np.flipud(sw_fun_E[:-1, :]),
                                   _np.int(N_ph / 2), axis=1)))

    # extend to full domain
    sw_fun_E = _np.vstack((sw_fun_E, -sw_fun_E[1:-1, :]))
    return sw_fun_E


def _main():
    # Test function extension

    ########
    # Integer
    ########
    s = 2
    int_fun = lambda th, ph: (3 * _np.exp(-1j * ph) *
                         (3 * _np.sin(th) + 2 * _np.sin(2 * th) +
                          7 * _np.sin(3 * th) - 7 * _np.sin(4 * th)) /
                         (32 * _np.sqrt(2 * _np.pi)))

    # Evaluate on grid [This requires an even number of samples]
    N_th, N_ph = 4, 6

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Extended domain lattice
    int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_E))
    int_ph = 2 * _np.pi / N_ph * _np.arange(0, (N_ph - 1) + 1)

    # We sample the function on the extended domain directly
    int_fun_N = _np.zeros((N_th_E, N_ph), dtype=_np.complex128)

    for i in _np.arange(0, N_th_E):
        for j in _np.arange(0, N_ph):
            int_fun_N[i, j] = int_fun(int_th[i], int_ph[j])

    # Throw small values
    int_fun_N[_np.abs(int_fun_N) < 1e-14] = 0

    # Restrict to fundamental (S2) domain
    int_fun_N_S2 = int_fun_N[0:N_th, :]

    # Perform extension
    int_fun_N_ext = int_funcExtension(s, int_fun_N_S2)

    # Check difference in extension and full analytical specification on torus
    diff_int_ext = _np.max(_np.abs(int_fun_N_ext - int_fun_N))

    #############
    # Half-integer
    #############
    h_s = -3 / 2
    h_int_fun = lambda th, ph: (1 / 40 * _np.exp(-3 / 2 * 1j * ph) *
                           _np.sqrt(3 / _np.pi) *
                           _np.cos(th / 2) * (
                               -20 * _np.exp(2 * 1j * ph) *
                               (-1 + _np.cos(th)) +
                               11 * _np.sqrt(2) *
                               _np.cos(th / 2)**2 * (-3 + 5 * _np.cos(th))))

    # Evaluate on grid [here we put an even number of samples]
    # We will however pass an _odd_ number of samples to the extension fcn.
    h_N_th, h_N_ph = 4, 6

    # Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2 * (h_N_th - 1), 4 * (h_N_th - 1)
    h_N_ph_E = 2 * h_N_ph

    # Extended domain lattice
    h_int_th = _np.pi / (h_N_th - 1) * _np.arange(0, (h_N_th_EE - 1) + 1)
    h_int_ph = 2 * _np.pi / h_N_ph * _np.arange(0, h_N_ph_E)

    # We sample the function on the extended domain directly
    h_int_fun_N = _np.zeros((h_N_th_EE, h_N_ph_E), dtype=_np.complex128)
    for i in _np.arange(0, h_N_th_EE):
        for j in _np.arange(0, h_N_ph_E):
            h_int_fun_N[i, j] = h_int_fun(h_int_th[i], h_int_ph[j])

    # Restrict to fundamental (S2) domain
    h_int_fun_N_S2 = h_int_fun_N[0:h_N_th, 0:h_N_ph]

    # Perform extension
    h_int_fun_N_ext = h_int_funcExtension(h_s, h_int_fun_N[0:h_N_th, 0:h_N_ph])

    # Check difference in extension and full analytical specification on torus
    diff_h_int_ext = _np.max(_np.abs(h_int_fun_N_ext - h_int_fun_N))

if __name__ == '__main__':
    _main()

#
# :D
#
