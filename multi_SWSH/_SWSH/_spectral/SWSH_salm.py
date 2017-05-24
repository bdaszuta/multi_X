"""
 ,-*
(_) Created on Tue Dec 29 16:52:16 2015

@author: Boris Daszuta
@function: Computation of coefficients.
"""
import numpy as _np
import numba as _nu

from multi_SWSH._SWSH._spectral import SWSH_Imn
from multi_SWSH._SWSH._special import wigner_d_TN
from multi_SWSH._types import (_COMPLEX_PREC, _INT_PREC)
from multi_SWSH._settings import _JIT_KWARGS


@_nu.jit(**_JIT_KWARGS)
def _int_salm(l, l_sqFa, n_ma, m_ma, s_arr, K_mn, int_Del_quad, salm_arr):
    '''
    Jit compilation of loop for Del calculation
    '''
    num_fun = salm_arr.shape[0]

    # Loop limits
    m_lim = _np.zeros(2, dtype=_INT_PREC)
    m_lim[0], m_lim[1] = n_ma, l
    m_lim = _np.min(m_lim) + 1

    q_lim = _np.zeros(2, dtype=_INT_PREC)
    q_lim[0], q_lim[1] = m_ma, l
    q_lim = _np.min(q_lim) + 1

    # ph_s = (1j)**(s_arr)
    # ph_m = (1j)**(-_np.arange(0, m_lim))

    # take care of positive m
    for m in _np.arange(0, m_lim):
        # Index for salm
        ind_lm = l * (l + 1) + m

        for f in _np.arange(0, num_fun):
            if _np.abs(s_arr[f]) <= l:
                for q in _np.arange(0, q_lim):

                    # use symmetries to treat negative spin-weight terms
                    if s_arr[f] < 0:
                        # sp_ph = (-1)**(l-q)
                        sp_ph = (1 - 2 * _np.mod(l - q, 2))
                    else:
                        sp_ph = 1

                    del_sp = sp_ph * int_Del_quad[q, _np.abs(s_arr[f])]

                    salm_arr[f, ind_lm] += \
                        int_Del_quad[q, m] * K_mn[f, q, n_ma + m] * del_sp

                # Apply scaling and phase
                salm_arr[f, ind_lm] = \
                    salm_arr[f, ind_lm] * l_sqFa * (1j**(s_arr[f] - m))

                # salm_arr[f,ind_lm] = \
                #    salm_arr[f,ind_lm]*l_sqFa*(ph_s[f]*1j**(-m))

    # take care of negative m
    for m in _np.arange(-m_lim + 1, 0):
        # Index for salm
        ind_lm = l * (l + 1) + m

        for f in _np.arange(0, num_fun):
            if _np.abs(s_arr[f]) <= l:
                for q in _np.arange(0, q_lim):

                    # use symmetries to treat negative spin-weight terms
                    if s_arr[f] < 0:
                        # sp_ph = (-1)**(l-q)
                        sp_ph = (1 - 2 * _np.mod(l - q, 2))
                    else:
                        sp_ph = 1

                    del_sp = sp_ph * int_Del_quad[q, _np.abs(s_arr[f])]

                    # salm_arr[f,ind_lm] += \
                    #    (-1)**(l-q)*int_Del_quad[q,-m]* \
                    #    K_mn[f,q,n_ma+m]*del_sp
                    salm_arr[f, ind_lm] += \
                        (1 - 2 * _np.mod(l - q, 2)) * int_Del_quad[q, -m] * \
                        K_mn[f, q, n_ma + m] * del_sp

                # apply scaling and phase
                salm_arr[f, ind_lm] = \
                    salm_arr[f, ind_lm] * l_sqFa * (1j**(s_arr[f] - m))


def int_salm(s_arr, f_arr):
    '''
    Arrays are expected as i_nput (see interface)
    '''
    # [If leaks occur] Prevent memory issues when called from external library
    # s_arr = _np.copy(s_arr)
    # f_arr = _np.copy(f_arr)

    # get total number of i_nput functions
    num_fun = (f_arr.shape)[0]

    # infer extension sampling
    N_th = f_arr[0].shape[0]
    N_th_E = 2 * (N_th - 1)
    N_ph = f_arr[0].shape[1]
    L_ph = _INT_PREC((N_ph - 2) / 2)

    # computation of weights
    n_wei = SWSH_Imn.int_wei_th(N_th_E)

    # calculate I_mn for each function
    I_mn_arr = _np.zeros((num_fun, N_th_E - 1, N_ph - 1),
                         dtype=_COMPLEX_PREC)
    for f in _np.arange(num_fun):
        I_mn_arr[f] = \
            SWSH_Imn.int_quadrature(s_arr[f], f_arr[f], n_wei)[1:, 1:]

    # infer I_mn dimensions and hence maximal m,n values
    m_ma, n_ma = _INT_PREC((N_th_E - 2) / 2), _INT_PREC((N_ph - 2) / 2)

    # prepare Kmn
    K_mn = I_mn_arr[:, m_ma:, :]

    for f in _np.arange(num_fun):
        # apply symmetry
        # I_mn_ph = _np.array([(-1)**(_np.arange(-n_ma, n_ma+1)+s_arr[f])])
        I_mn_ph = (
            _np.array([(1 - 2 *
                        _np.mod(_np.arange(-n_ma, n_ma + 1) + s_arr[f], 2))]))

        K_mn[f, 1:, :] += (
            _np.repeat(I_mn_ph, m_ma, axis=0) *
            _np.flipud(I_mn_arr[f, :m_ma, :]))

    # infer maximal required Del element
    L = _INT_PREC((N_th_E - 6) / 4)

    # assign space for coefficients
    salm_arr = _np.zeros((num_fun, (L + 1)**2), dtype=_COMPLEX_PREC)

    # compute del face
    int_Del_strip = wigner_d_TN.int_Del_Strip(L)
    int_Del_ext = wigner_d_TN.int_Del_Exterior(L, int_Del_strip, L_ph)

    for l in _np.arange(_np.min(_np.abs(s_arr)), L + 1):
        l_sqFa = _np.sqrt((2 * l + 1) / (4 * _np.pi))

        # compute current internal Del values
        int_Del_int = wigner_d_TN.int_Del_Interior(l, int_Del_ext, L_ph)
        int_Del_quad = wigner_d_TN.int_Del_Interior_ExtendQuad(
            l, int_Del_int, L_ph)

        _int_salm(l, l_sqFa, n_ma, m_ma,
                  _INT_PREC(s_arr), K_mn, int_Del_quad, salm_arr)

    return salm_arr


@_nu.jit(**_JIT_KWARGS)
def _h_int_salm(l2, l_sqFa, n_ma, m_ma, I_mn_m_ind, I_mn_n_ind, s_arr, J_mn,
                h_int_Del_quad, salm_arr):
    '''
    Jit compilation of loop for Del calculation
    '''
    num_fun = salm_arr.shape[0]

    # loop limits
    m_lim = _np.zeros(2, dtype=_INT_PREC)
    m_lim[0], m_lim[1] = n_ma, l2
    # increase for correct arange maxima
    m_lim = _np.min(m_lim) + 1

    q_lim = _np.zeros(2, dtype=_INT_PREC)
    q_lim[0], q_lim[1] = m_ma, l2
    # increase for correct arange maxima
    q_lim = _np.min(q_lim) + 1

    # take care of positive m
    for m2 in _np.arange(1, m_lim, 2):
        # index for salm
        ind_lm = _INT_PREC(1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in _np.arange(0, num_fun):
            if _INT_PREC(_np.abs(2 * s_arr[f])) <= l2:
                for q2 in _np.arange(1, q_lim, 2):
                    # use symmetries to treat negative spin-weight terms
                    if s_arr[f] < 0:
                        # sp_ph = (-1)**((l2-q2)/2)
                        sp_ph = (1 - 2 * _np.mod((l2 - q2) / 2, 2))
                    else:
                        sp_ph = 1

                    # array indices
                    q_i = _INT_PREC((q2 - 1) / 2)
                    s_i_f = _INT_PREC((_np.abs(s_arr[f] * 2) - 1) / 2)
                    m_i = _INT_PREC((m2 - 1) / 2)

                    del_sp = sp_ph * h_int_Del_quad[q_i, s_i_f]

                    salm_arr[f, ind_lm] += \
                        h_int_Del_quad[q_i, m_i] * \
                        J_mn[f, q_i, _INT_PREC(m_i + I_mn_n_ind)] * del_sp

                # apply scaling and phase
                salm_arr[f, ind_lm] = \
                    salm_arr[f, ind_lm] * l_sqFa * (1j**(s_arr[f] - m2 / 2))

    # take care of negative m
    for m2 in _np.arange(-m_lim + 1, 1, 2):
        # index for salm
        ind_lm = _INT_PREC(1 / 2 * (m2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in _np.arange(0, num_fun):
            if _INT_PREC(_np.abs(2 * s_arr[f])) <= l2:
                for q2 in _np.arange(1, q_lim, 2):
                    # use symmetries to treat negative spin-weight terms
                    if s_arr[f] < 0:
                        # sp_ph = (-1)**((l2-q2)/2)
                        sp_ph = (1 - 2 * _np.mod((l2 - q2) / 2, 2))
                    else:
                        sp_ph = 1

                    # array indices
                    q_i = _INT_PREC((q2 - 1) / 2)
                    s_i_f = _INT_PREC((_np.abs(s_arr[f] * 2) - 1) / 2)
                    m_i = _INT_PREC((_np.abs(m2) - 1) / 2)

                    del_sp = sp_ph * h_int_Del_quad[q_i, s_i_f]

                    # salm_arr[f,ind_lm] += \
                    #    (-1)**((l2-q2)/2)*h_int_Del_quad[q_i,m_i]* \
                    #    J_mn[f,q_i,_INT_PREC(I_mn_n_ind-m_i-1)]*del_sp

                    salm_arr[f, ind_lm] += (
                        (1 - 2 * _np.mod((l2 - q2) / 2, 2)) *
                        h_int_Del_quad[q_i, m_i] *
                        J_mn[f, q_i, _INT_PREC(I_mn_n_ind - m_i - 1)] * del_sp)

                # apply scaling and phase
                salm_arr[f, ind_lm] = \
                    salm_arr[f, ind_lm] * l_sqFa * \
                    (1j**(s_arr[f] + _np.abs(m2) / 2))


def h_int_salm(s_arr, f_arr):
    '''
    Arrays are expected as i_nput (see interface)
    '''
    # get total number of i_nput functions
    num_fun = (f_arr.shape)[0]

    # infer extension sampling
    h_N_th, h_N_ph = f_arr.shape[1], f_arr.shape[2]

    h_L_ph = _INT_PREC((h_N_ph - 1))

    # extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2 * (h_N_th - 1), 4 * (h_N_th - 1)
    h_N_ph_E = 2 * h_N_ph

    # computation of weights
    h_n_wei_th, h_n_wei_ph = SWSH_Imn.h_int_wei(h_N_th_EE, h_N_ph_E)

    dImn_m = _INT_PREC((h_N_th_E - 2 - h_N_th) / 2)

    # calculate I_mn for each function
    I_mn_arr = _np.zeros((num_fun, h_N_th, h_N_ph),
                         dtype=_COMPLEX_PREC)
    for f in _np.arange(num_fun):
        I_mn_arr[f] = SWSH_Imn.h_int_quadrature(
            s_arr[f], f_arr[f],
            h_n_wei_th, h_n_wei_ph)[dImn_m:-dImn_m, :]

    # infer I_mn dimensions and hence maximal m,n values
    # [Integral part to be divided by 2]
    m_ma, n_ma = h_N_th - 1, h_N_ph - 1

    # index for m=1/2
    I_mn_m_ind = _INT_PREC(h_N_th / 2)
    I_mn_n_ind = _INT_PREC(h_N_ph / 2)

    # prepare Jmn
    J_mn = I_mn_arr[:, I_mn_m_ind:, :]

    for f in _np.arange(num_fun):
        # compute phase
        pwr_v = _INT_PREC(_np.arange(-n_ma, n_ma + 1, 2) / 2 + s_arr[f])
        I_mn_ph = -_np.array([(-1)**pwr_v])

        I_mn_ph = _np.repeat(I_mn_ph, I_mn_m_ind, axis=0)

        J_mn[f, :, :] += \
            I_mn_ph * _np.flipud(I_mn_arr[f, :I_mn_m_ind, :])

    # return J_mn

    # m_ma also functions as the maximal required Del element
    L2 = m_ma

    # assign space for coefficients
    sz_salm = _INT_PREC(3 / 4 + (L2 / 2) * (2 + L2 / 2))
    salm_arr = _np.zeros((num_fun, sz_salm), dtype=_COMPLEX_PREC)

    # compute del face
    h_int_Del_strip = wigner_d_TN.h_int_Del_Strip(L2)
    h_int_Del_ext = wigner_d_TN.h_int_Del_Exterior(L2, h_int_Del_strip, h_L_ph)

    for l2 in _np.arange(_np.min(_np.abs(2 * s_arr)), L2 + 1, 2):
        l_sqFa = _np.sqrt((l2 + 1) / (4 * _np.pi))

        # Compute current internal Del values
        h_int_Del_int = wigner_d_TN.h_int_Del_Interior(
            l2, h_int_Del_ext, h_L_ph)
        h_int_Del_quad = wigner_d_TN.h_int_Del_Interior_ExtendQuad(
            l2, h_int_Del_int, h_L_ph)

        _h_int_salm(l2, l_sqFa, n_ma, m_ma, I_mn_m_ind, I_mn_n_ind, s_arr,
                    J_mn, h_int_Del_quad, salm_arr)

    return salm_arr

#
# :D
#
