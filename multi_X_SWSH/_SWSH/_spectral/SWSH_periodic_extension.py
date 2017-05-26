"""
 ,-*
(_) Created on Sat Dec 19 15:36:55 2015

@author: Boris Daszuta
@function: Periodic extension mechanism for spin-weighted functions.
"""
import numpy as _np

from multi_X_SWSH._types import (_INT_PREC, _REAL_PREC)


def int_funcExtension(s, sw_fun):
    # [0,pi]x[0,2pi] -> [0,2pi]x[0,2pi]
    N_ph = sw_fun.shape[1]
    return _np.vstack(
        (sw_fun, (-1) ** _REAL_PREC(s) *
         _np.flipud(_np.roll(sw_fun[1:-1, :],
                             N_ph // 2, 1))))


def int_funcDoubleExtension(s, sw_fun):
    sw_fun_T2 = int_funcExtension(s, sw_fun)

    # now extend again:
    sw_fun_T2_E = _np.repeat(sw_fun_T2, 2, axis=1)
    return _np.repeat(sw_fun_T2_E, 2, axis=0)


def h_int_funcExtension(s, sw_fun):
    # [0,pi]x[0,2pi] -> [0,4pi]x[0,4pi]
    N_th, N_ph = sw_fun.shape

    # extend to D_I U D_II
    sw_fun_E = _np.hstack((sw_fun, -sw_fun))

    s_int = _INT_PREC(2 * s)
    s_ph = -(1j ** s_int)

    # extend to D_III U D_IV
    sw_fun_E = _np.vstack(
        (sw_fun_E, s_ph * _np.roll(_np.flipud(sw_fun_E[:-1, :]),
                                   N_ph // 2, axis=1)))

    # extend to full domain
    sw_fun_E = _np.vstack((sw_fun_E, -sw_fun_E[1:-1, :]))
    return sw_fun_E


def sf_periodic_extension(s=None, sf=None, is_half_integer=True,
                          double_extension=False):
    '''
    Periodically extend a (half)-integer spin-weighted function on the
    2-sphere to the 2-torus. Optionally perform double-extension.

    Parameters
    ----------
    s = None : int
        Spin-weight.

    sf = None : array-like
        Spin-weighted function.

    is_half_integer = True : bool
        Control whether input corresponds to (half)-integer spin-weight.

    double_extension = False :bool
        Control whether double extension is to be performed.

    Notes
    -----
    An even number of samples is required for each extension.
    '''
    if is_half_integer:
        sw_fun_E = h_int_funcExtension(s, sf)
    else:
        sw_fun_E = int_funcExtension(s, sf)

    if double_extension:
        sw_fun_E = _np.repeat(sw_fun_E, 2, axis=1)
    return sw_fun_E


#
# :D
#
