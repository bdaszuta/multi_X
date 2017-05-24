"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience function for the construction and evaluation of the
action of the eth operator on spin-weighted functions in the coefficient space
representation.
"""
import numba as _nu
import numpy as _np
import fastcache as _fc

from multi_SWSH._types import (_INT_PREC, _REAL_PREC)
from multi_SWSH._settings import (_JIT_KWARGS, _FC_KWARGS)


@_fc.clru_cache(**_FC_KWARGS)
@_nu.jit(**_JIT_KWARGS)
def eth_build(s=None, L=None, type=-1, is_half_integer=True):
    '''
    Construct representation of eth operator in coefficient space.

    Parameters
    ----------
    s = None : int
        The spin-weight we currently have.

    L = None : int
        The 'th' band-limit being used.

    type = -1 : -1 (or +1)
        Control whether raising or lowering operator is being constructed.

    is_half_integer = True : bool
        Control whether values are constructed that correspond to (half)
        integer spin-weight.

    Returns
    -------
    array-like
        Array with \mp sqrt((l \mp s) (l \pm s + 1)))

    Notes
    -----
    Construction of operator representation is cached.
    '''
    if is_half_integer:
        arr_sz = _INT_PREC(3 / 4 + (L / 2) * (2 + L / 2))
        eth_op = _np.zeros(arr_sz, dtype=_REAL_PREC)
        for l in _np.arange(_np.abs(s), L + 1, 2):
            for m in _np.arange(-L, L + 1, 2):
                idx = _INT_PREC(((m + l * (l / 2 + 1) - 1 / 2)) / 2)
                # map to half integer differences
                lps, lms = (l + s) / 2, (l - s) / 2
                if type == -1:
                    eth_op[idx] = _np.sqrt(lps * (lms + 1))
                else:
                    eth_op[idx] = -_np.sqrt(lms * (lps + 1))
    else:
        arr_sz = (L + 1) ** 2
        eth_op = _np.zeros(arr_sz, dtype=_REAL_PREC)
        for l in _np.arange(_np.abs(s), L + 1):
            for m in _np.arange(-L, L + 1):
                idx = l * (l + 1) + m
                if type == -1:
                    eth_op[idx] = _np.sqrt((l + s) * (l - s + 1))
                else:
                    eth_op[idx] = -_np.sqrt((l - s) * (l + s + 1))

    return eth_op

#
# :D
#
