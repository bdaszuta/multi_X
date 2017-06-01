"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience functions for calculation of array sizes based on
desired band-limit and conversions between 1d array idxs and (l, m) type
indices.
"""
import numpy as _np
import numba as _nu

from multi_X_SWSH._types import (_INT_PREC, _COMPLEX_PREC)
from multi_X_SWSH._settings import _JIT_KWARGS
from multi_X_SWSH._doc_replacements import _rep_doc


def arr_idx_map(l=None, m=None, is_half_integer=True):
    '''
    Map a pair ($m_l, $m_m) to a one-dimensional array index.

    Parameters
    ----------
    l = None : $int (or $arr)
        The $m_l mode.

    m = None : $int (or $arr)
        The $m_m mode.

    is_half_integer = True : $bool
        $is_half_integer

    Returns
    -------
    $int (or $arr)
        The one-dimensional idx for accessing the requisite entry.
    '''
    # as usual l, m should be _integer_ in both cases.
    if is_half_integer:
        return _INT_PREC(((m + l * (l / 2 + 1) - 1 / 2)) / 2)
    return l * (l + 1) + m


@_nu.jit(**_JIT_KWARGS)
def arr_sz_calc(L_th=None, is_half_integer=True):
    '''
    Calculate the size of an array based on band-limit.

    Parameters
    ----------
    L_th = None : $int
        Band-limit.

    is_half_integer = True : bool
        $is_half_integer

    Returns
    -------
    $int
        The size of the array.
    '''
    # calculate the required size of an array based on band-limit
    if is_half_integer:
        return _INT_PREC(3 / 4 + (L_th / 2) * (2 + L_th / 2))
    return (L_th + 1) ** 2


@_nu.jit(**_JIT_KWARGS)
def L_to_N(L_th=None, L_ph=None, is_half_integer=True):
    '''
    Convenience function for converting band-limits to the number of samples;
    assume Nyquist.

    Parameters
    ----------
    L_th = None : $int
        $L_th

    L_ph = None : $int
        (optional) $L_ph

    is_half_integer = True : $bool
        $is_half_integer

    Returns
    -------
    tuple : (N_th, N_ph)

    See also
    --------
    N_to_L : opposite direction.
    '''
    if L_ph is None:
        L_ph = L_th

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


def L_compatible(L_th=None, L_ph=None, is_half_integer=True):
    '''
    Convenience function to infer compatible band-limits for sampling of
    (half)-integer spin-weighted fields.

    For input half-integer band-limits (note that we take only the numerator):

    .. math::

        \\begin{equation}
            (L_{\\vartheta},\,L_{\\varphi}) \mapsto
            ((L_{\\vartheta} - 3) / 2,\, (L_{\\varphi} - 1) / 2);
        \\end{equation}

    Parameters
    ----------
    L_th = None : int
        The band-limit in the 'th' direction to use.

    L_ph = None : int
        The band-limit in the 'ph' direction to use.

    is_half_integer = True : bool
        Specify what input 'L_th' and 'L_ph' band-limits correspond to.

    Returns
    -------
    tuple : (L_th, L_ph)

    Examples
    --------
    >>> L_compatible(L_th=33, L_ph=33, is_half_integer=True)
    (15, 16)

    >>> L_compatible(L_th=32, L_ph=32, is_half_integer=False)
    (67, 65)

    See also
    --------
    build_grid
    '''
    if is_half_integer:
        return _INT_PREC((L_th - 3) // 2), _INT_PREC((L_ph - 1) // 2)
    return (_INT_PREC(3 + 2 * L_th), _INT_PREC(1 + 2 * L_ph))


def salm_from_idx_dict(idx_dict=None, L_th=None, L_ph=None, is_dense=True,
                       is_half_integer=True):
    '''
    Convenience function for construction of an salm array via a dictionary.

    Parameters
    ----------
    idx_dict = None : dict
        Dictionary to use during 'salm' construction. Keys should correspond to
        a (l, m) tuple with associated data being the value that is to be
        inserted.

    L_th = None : int
        The band-limit to use in the 'th' direction.

    L_ph = None : int
        (Optional) The band-limit to use in the 'ph' direction.

    is_dense = True : bool
        Control whether a sparse 'salm' array is to be constructed.

    is_half_integer = True : bool
        Specify what type of parameters are in use.

    Returns
    -------
    array-like
        The populated array

    Examples
    --------
    >>> salm_from_idx_dict(idx_dict={(1, 1): 1, (3, 1): -1}, L_th=5,
    ...                    is_half_integer=True)  # doctest: +SKIP
    array([ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,
            0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j])
    '''
    if L_ph is None:
        L_ph = L_th

    salm_sz = arr_sz_calc(L_th=L_th, is_half_integer=is_half_integer)
    salm = _np.zeros(salm_sz, dtype=_COMPLEX_PREC)

    for lm_tup, val in idx_dict.items():
        ix = arr_idx_map(l=lm_tup[0], m=lm_tup[1],
                         is_half_integer=is_half_integer)
        salm[ix] = val

    return salm

###############################################################################
# inject doc vars.
_rep_doc(arr_idx_map)
_rep_doc(arr_sz_calc)
_rep_doc(L_to_N)
###############################################################################

#
# :D
#
