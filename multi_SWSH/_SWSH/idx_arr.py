"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience functions for calculation of array sizes based on
desired band-limit and conversions between 1d array idxs and (l, m) type
indices.
"""
import numba as _nu

from multi_SWSH._types import _INT_PREC
from multi_SWSH._settings import _JIT_KWARGS


def arr_idx_map(l=None, m=None, is_half_integer=True):
    '''
    Map a pair (`l`, `m`) to a one-dimensional array index.

    Parameters
    ----------
    l = None : int (or array-like)
        The 'l' mode.

    m = None : int (or array-like)
        The 'm' mode.

    Returns
    -------
    int (or array-like)
        The one-dimensional idx for accessing the requisite entry.
    '''
    # as usual l, m should be _integer_ in both cases.
    if is_half_integer:
        return _INT_PREC(((m + l * (l / 2 + 1) - 1 / 2)) / 2)
    return l * (l + 1) + m


@_nu.jit(**_JIT_KWARGS)
def arr_sz_calc(L=None, is_half_integer=True):
    '''
    Calculate the size of an array based on band-limit.

    Parameters
    ----------
    L = None : int
        Band-limit.

    is_half_integer = True : bool
        Control whether values are constructed that correspond to (half)
        integer spin-weight.

    Returns
    -------
    int
        The size of the array.
    '''
    # calculate the required size of an array based on band-limit
    if is_half_integer:
        return _INT_PREC(3 / 4 + (L / 2) * (2 + L / 2))
    return (L + 1) ** 2


@_nu.jit(**_JIT_KWARGS)
def L_to_N(L_th=None, L_ph=None, is_half_integer=True):
    '''
    Convenience function for converting band-limits to the number of samples;
    assume Nyquist.

    Parameters
    ----------
    L_th = None : int
        The band-limit in the 'th' direction to use.

    L_ph = None : int
        The band-limit in the 'ph' direction to use.

    is_half_integer = True : bool
        Control whether arrays are constructed that correspond to (half)
        integer spin-weight.

    Returns
    -------
    tuple : (N_th, N_ph)

    See also
    --------
    N_to_L : opposite direction.
    '''
    if is_half_integer:
        L_th, L_ph = L_th / 2, L_ph / 2
        return (_INT_PREC(2 * L_th + 1), _INT_PREC(2 * L_ph + 1))

    return _INT_PREC(2 * (L_th + 2)), _INT_PREC(2 * L_ph + 2)


@_nu.jit(**_JIT_KWARGS)
def N_to_L(N_th=None, N_ph=None, is_half_integer=True):
    '''
    Convenience function for converting number of samples to band-limits;
    assume Nyquist.

    Parameters
    ----------
    N_th = None : int
        The number of samples in the 'th' direction to use.

    N_ph = None : int
        The number of samples in the 'ph' direction to use.

    is_half_integer = True : bool
        Control whether arrays are constructed that correspond to (half)
        integer spin-weight.

    Returns
    -------
    tuple : (L_th, L_ph)

    Examples
    --------
    >>> N_to_L(11, 11, is_half_integer=True)
    (10, 10)

    See also
    --------
    L_to_N : opposite direction.
    '''
    if is_half_integer:
        return (N_th - 1), (N_ph - 1)
    return _INT_PREC(N_th / 2 - 2), _INT_PREC((N_ph - 2) / 2)

#
# :D
#
