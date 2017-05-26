"""
 ,-*
(_) Created on <Tue May 23 2017> @ 14:20:20

@author: Boris Daszuta
@function: Convenience function for construction of a numerical grid based on a
supplied band-limit.
"""
import numpy as _np
import numba as _nu

from multi_X_SWSH._SWSH.idx_arr import L_to_N
from multi_X_SWSH._settings import _JIT_KWARGS
from multi_X_SWSH._doc_replacements import _rep_doc


@_nu.jit(**_JIT_KWARGS)
def build_grid(L_th=None, L_ph=None, is_half_integer=True, is_extended=False,
               is_double_extension=False):
    '''
    Construct $m_th, $m_ph one dimensional grids with optional extension from
    $m_S2; appropriate sampling $m_N_th, $m_N_ph inferred automatically.

    Parameters
    ----------
    L_th = None : $int
        $L_th

    L_ph = None : $int
        $L_ph

    is_half_integer = True : $bool
        $is_half_integer

    is_extended = False : $bool
        $is_extended

    is_double_extension = False : $bool
        $is_double_extension

    Returns
    -------
    (th, ph) : ($arr, $arr)
        The constructed grids.

    Examples
    --------
    >>> build_grid(L_th=3, L_ph=5, is_half_integer=True)  # doctest: +SKIP
    (array([ 0.        ,  1.04719755,  2.0943951 ,  3.14159265]),
     array([ 0.        ,  1.04719755,  2.0943951 ,  3.14159265,  4.1887902 ,
             5.23598776]))

    Automatic periodic extension:

    >>> build_grid(L_th=2, L_ph=2, is_half_integer=False,
    ...            is_extended=True)  # doctest: +SKIP
    (array([ 0.        ,  0.44879895,  0.8975979 ,  1.34639685,  1.7951958 ,
             2.24399475,  2.6927937 ,  3.14159265,  3.5903916 ,  4.03919055,
             4.48798951,  4.93678846,  5.38558741,  5.83438636]),
     array([ 0.        ,  1.04719755,  2.0943951 ,  3.14159265,  4.1887902 ,
             5.23598776]))
    '''
    N_th, N_ph = L_to_N(L_th=L_th, L_ph=L_ph, is_half_integer=is_half_integer)

    if is_half_integer:
        # extension in th,ph requires more points
        N_th_E, N_th_EE = 2 * (N_th - 1), 4 * (N_th - 1)
        N_ph_E = 2 * N_ph

        # extended domain lattice
        h_int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_EE - 1) + 1)
        h_int_ph = 2 * _np.pi / N_ph * _np.arange(0, N_ph_E)

        if is_extended:
            if is_double_extension:
                return (_np.hstack((h_int_th, h_int_th + 4 * _np.pi)),
                        _np.hstack((h_int_ph, h_int_ph + 4 * _np.pi)))
            return h_int_th, h_int_ph

        return h_int_th[:N_th], h_int_ph[:N_ph]

    # extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # extended domain lattice
    int_th = _np.pi / (N_th - 1) * _np.arange(0, (N_th_E))
    int_ph = 2 * _np.pi / N_ph * _np.arange(0, (N_ph - 1) + 1)

    if is_extended:
        if is_double_extension:
            return (_np.hstack((int_th, int_th + 2 * _np.pi)),
                    _np.hstack((int_ph, int_ph + 2 * _np.pi)))
        return int_th, int_ph
    return int_th[:N_th], int_ph[:N_ph]

###############################################################################
# inject doc vars.
_rep_doc(build_grid)
###############################################################################

#
# :D
#
