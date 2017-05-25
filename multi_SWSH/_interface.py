"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:11:23

@author: Boris Daszuta
@function: Interface for making use of the SWSH transformations;
see read.nfo.
"""
import numpy as _np
# import numba as _nu

from multi_SWSH import (_SWSH, _types)
from multi_SWSH._SWSH.random_coefficients import (_int_gen_rand_salm,
                                                  _h_int_gen_rand_salm)


def _int_salm(s_arr, f_arr):
    # Compute salm from integer spin-weighted functions sampled on S2

    # ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_types._INT_PREC)
    if len(f_arr.shape) == 2:
        f_arr = _np.array([f_arr], dtype=_types._COMPLEX_PREC)
    return _SWSH.SWSH_salm.int_salm(s_arr, f_arr)


def _h_int_salm(s_arr, f_arr):
    # Compute salm from half-integer spin-weighted functions sampled on S2

    s_arr = s_arr / 2

    # Ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_types._REAL_PREC)
    if len(f_arr.shape) == 2:
        f_arr = _np.array([f_arr], dtype=_types._COMPLEX_PREC)

    return _SWSH.SWSH_salm.h_int_salm(s_arr, f_arr)


def _int_sf(s_arr, salm_arr_i, N_th, N_ph):
    # compute integer spin-weighted functions on S2

    salm_arr = _np.copy(salm_arr_i)
    # Ensure we deal with arrays
    if _np.isscalar(s_arr):
        s_arr = _np.array([s_arr], dtype=_types._REAL_PREC)
    if len(salm_arr.shape) == 1:
        salm_arr = _np.array([salm_arr], dtype=_types._COMPLEX_PREC)

    return _SWSH.SWSH_bwd.int_sf(s_arr, salm_arr, N_th, N_ph)


def _h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph):
    # compute half-integer spin-weighted functions on S2

    h_s_arr = h_s_arr / 2

    # Ensure we deal with arrays
    h_s_arr = _np.asanyarray(h_s_arr)
    h_s_arr = _np.resize(h_s_arr, h_s_arr.size)

    if len(h_salm_arr.shape) == 1:
        h_salm_arr = _np.array([h_salm_arr], dtype=_types._COMPLEX_PREC)

    return _SWSH.SWSH_bwd.h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph)


def generate_random_salm(s=None, L_th=None, L_ph=None, L_th_pad=None,
                         is_half_integer=True):
    '''
    Convenience function for construction of (uniform) random coefficients
    with:

    .. math::

        \mathfrak{R}({}_sa_{lm}),\, \mathfrak{I}({}_sa_{lm}) \in [-1,\,1]

    Parameters
    ----------
    s = None : int (array-like)
        Spin-weights to construct arrays for.

    L_th = None : int
        Band-limit (in :math:`\\vartheta`) to use.

    L_ph = None : int
        (Optional) band-limit (in :math:`\\varphi`) to use.

    L_th_pad = None : int
        (Optional) Control whether the values are padded to some higher
        band-limit.

    is_half_integer = True : bool
        Control whether arrays are constructed that correspond to (half)
        integer spin-weight.

    Returns
    -------
    Stacked array (first dimension controlled by `s` as described above) of
    random values.
    '''
    s = _np.array(s)
    s = _np.resize(s, s.size)

    if is_half_integer:
        return _h_int_gen_rand_salm(s, L_th, L_ph, L_th_pad)
    return _int_gen_rand_salm(s, L_th, L_ph, L_th_pad)


def sf_to_salm(s=None, f=None, is_half_integer=True):
    '''
    Transform input spin-weighted functions to coefficients.

    Parameters
    ----------
    s = None : int (or array-like)
        The spin weight(s).

    f = None : array-like
        The functions to transform.

    is_half_integer = True : bool
        Control the type of transformation.

    Returns
    -------
    array-like
        Coefficients.
    '''
    if is_half_integer:
        return _h_int_salm(s, f)
    return _int_salm(s, f)


def salm_to_sf(s=None, alm=None, N_th=None, N_ph=None, is_half_integer=True):
    '''
    Transform input coefficients to spin-weighted functions.

    Parameters
    ----------
    s = None : int (or array-like)
        The spin weight(s).

    alm = None : array-like
        The coefficients to transform.

    N_th = None : int
        The number of samples in the 'th' direction to use.

    N_ph = None : int
        The number of samples in the 'ph' direction to use.

    is_half_integer = True : bool
        Control the type of transformation.

    Returns
    -------
    array-like
        Sampled functions.
    '''
    if is_half_integer:
        return _h_int_sf(s, alm, N_th, N_ph)
    return _int_sf(s, alm, N_th, N_ph)

#
# :D
#
