#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Tue Jan  5 22:52:50 2016

@author: Boris Daszuta
@function: Computation of function from coefficients representation

Note:
Please use int_sf and h_int_sf as these copy coefficient arrays and retain
the original

"""

import spectral.Interface_FourierSeries as FS
import special_functions.wigner_d_TN as wig

import numpy as np
import numba as nu
import joblib as jl


# test analytically defined sYlm
import special_functions.swsh_AnSum as swsh_an


num_prec_complex = np.complex128
uint_prec = np.uint64
int_prec = np.int64

# Fourier object instance
FSi = None

# For threading J_mn, K_mn
n_th = 1
par = jl.Parallel(n_jobs=n_th, backend='threading')
#par = jl.Parallel(n_jobs=n_th)


@nu.jit(nopython=True, nogil=True, cache=True)
def _int_J_mn_ph(s_arr, salm_arr, L_th, L_ph):
    '''
    Apply phase factors to salm coefficients
    '''
    num_fun = salm_arr.shape[0]

    ph_s = (1j)**(-s_arr)
    ph_n = (1j)**(-np.arange(-L_th, L_th + 1))

    # Mask coefficients with phases
    # for l in np.arange(np.min(np.abs(s_arr)), L_th+1):
    for l in np.arange(0, L_th + 1):
        l_sqFa = np.sqrt((2 * l + 1) / (4 * np.pi))
        l_fa = l * (l + 1)

        n_lim = L_ph if (L_ph <= l) else l

        for n in np.arange(-n_lim, n_lim + 1):
            ind_ln = l_fa + n
            for f in np.arange(0, num_fun):
                # if np.abs(s_arr[f])<=l:
                #ph = (1j)**(-s_arr[f]-n)
                ph = ph_s[f] * ph_n[L_th + n]

                salm_arr[f, ind_ln] = ph * l_sqFa * salm_arr[f, ind_ln]


#@nu.jit(nopython=True, nogil=True, cache=True)
def _int_J_mn_Del(l, n_lim, J_mn, m0_int, n0_int, int_Del_quad,
                  s_arr, salm_til_arr):
    '''
    Apply Del symmetries and compute J_mn functionals
    '''

    num_fun = salm_til_arr.shape[0]
    abs_s_arr = np.abs(s_arr)

    # Positive 'n' indices
    for n in np.arange(0, n_lim + 1):
        ind_ln = l * (l + 1) + n

        for f in np.arange(0, num_fun):
            if abs_s_arr[f] <= l:

                for m in np.arange(-l, 0):
                    prod_salmdel = salm_til_arr[f][ind_ln] * \
                        int_Del_quad[-m, abs_s_arr[f]] * int_Del_quad[-m, n]

                    # Negative 'm' indices
                    #sp_ph = (-1)**(l-m) if (s_arr[f] < 0) else 1

                    # J_mn[f,int_prec(m0+m), int_prec(n0+n)] += \
                    #    sp_ph * \
                    #    salm_til_arr[f][ind_ln]* \
                    #    int_Del_quad[-m,np.abs(s_arr[f])]* \
                    #    int_Del_quad[-m,n]

                    sp_ph = (1 - 2 * np.mod(l - m, 2)) if (s_arr[f] < 0) else 1

                    J_mn[f, m0_int + m, n0_int + n] += \
                        sp_ph * \
                        prod_salmdel

                    # Positive 'm' indices
                    #sp_ph = (-1)**(l+m) if (s_arr[f]<0) else 1
                    # J_mn[f,int_prec(m0-m), int_prec(n0+n)] += \
                    #    sp_ph*(-1)**(s_arr[f]+n)* \
                    #    salm_til_arr[f][ind_ln]* \
                    #    int_Del_quad[-m,np.abs(s_arr[f])]* \
                    #    int_Del_quad[-m,n]

                    sp_ph = (1 - 2 * np.mod(l + m + s_arr[f] + n, 2)) \
                        if (s_arr[f] < 0) else (1 - 2 * np.mod(s_arr[f] + n, 2))

                    J_mn[f, m0_int - m, n0_int + n] += \
                        sp_ph * \
                        prod_salmdel

                # m=0 case
                #sp_ph = (-1)**l if (s_arr[f]<0) else 1
                # J_mn[f, int_prec(m0), int_prec(n0+n)] += \
                #    sp_ph * \
                #    salm_til_arr[f][ind_ln]* \
                #    int_Del_quad[0,np.abs(s_arr[f])]* \
                #    int_Del_quad[0,n]
                sp_ph = (1 - 2 * np.mod(l, 2)) if (s_arr[f] < 0) else 1
                J_mn[f, m0_int, n0_int + n] += \
                    sp_ph * \
                    salm_til_arr[f][ind_ln] * \
                    int_Del_quad[0, abs_s_arr[f]] * \
                    int_Del_quad[0, n]

    # Negative 'n' indices
    for n in np.arange(-n_lim, 0):
        ind_ln = l * (l + 1) + n

        for f in np.arange(0, num_fun):
            if abs_s_arr[f] <= l:

                for m in np.arange(-l, 0):
                    # Negative 'm' indices
                    #sp_ph = (-1)**(l-m) if (s_arr[f]<0) else 1
                    prod_salmdel = salm_til_arr[f][ind_ln] * \
                        int_Del_quad[-m, abs_s_arr[f]] * int_Del_quad[-m, -n]

                    # J_mn[f,int_prec(m0+m), int_prec(n0+n)] += \
                    #    sp_ph*(-1)**(l+m)* \
                    #    salm_til_arr[f][ind_ln]* \
                    #    int_Del_quad[-m,np.abs(s_arr[f])]* \
                    #    int_Del_quad[-m,-n]
                    sp_ph = 1 \
                        if (s_arr[f] < 0) else (1 - 2 * np.mod(l + m, 2))

                    J_mn[f, m0_int + m, n0_int + n] += \
                        sp_ph * \
                        prod_salmdel

                    # Positive 'm' indices
                    #sp_ph = (-1)**(l+m) if (s_arr[f]<0) else 1
                    # J_mn[f,int_prec(m0-m), int_prec(n0+n)] += \
                    #    sp_ph*(-1)**(l-s_arr[f]-n-m)* \
                    #    salm_til_arr[f][ind_ln]* \
                    #    int_Del_quad[-m,np.abs(s_arr[f])]* \
                    #    int_Del_quad[-m,-n]
                    sp_ph = (1 - 2 * np.mod(-s_arr[f] - n, 2)) \
                        if (s_arr[f] < 0) else (1 - 2 * np.mod(l - s_arr[f] - n - m, 2))

                    J_mn[f, m0_int - m, n0_int + n] += \
                        sp_ph * \
                        prod_salmdel

                # m=0 case
                #sp_ph = (-1)**l if (s_arr[f]<0) else 1
                # J_mn[f, int_prec(m0), int_prec(n0+n)] += \
                #        sp_ph*(-1)**(l-s_arr[f]-n)* \
                #        salm_til_arr[f][ind_ln]* \
                #        int_Del_quad[0,np.abs(s_arr[f])]* \
                #        int_Del_quad[0,-n]
                sp_ph = (1 - 2 * np.mod(-s_arr[f] - n, 2)) \
                    if (s_arr[f] < 0) else (1 - 2 * np.mod(l - s_arr[f] - n, 2))

                J_mn[f, m0_int, n0_int + n] += \
                    sp_ph * \
                    salm_til_arr[f][ind_ln] * \
                    int_Del_quad[0, abs_s_arr[f]] * \
                    int_Del_quad[0, -n]


@nu.jit(nopython=True, nogil=True, cache=True)
def _int_J_mn_Del_opt(l, n_lim, J_mn, m0_int, n0_int, int_Del_quad,
                      s, salm_til_arr):
    '''
    Apply Del symmetries and compute J_mn functionals
    '''
    abs_s = np.abs(s)
    l_fa = l * (l + 1)

    # Pull phase computations outside loop
#    m_arr_1 = np.arange(-l,0)
#    n_arr_1 = np.arange(0, n_lim+1)
#    n_arr_2 = np.arange(-n_lim, 0)
#
#    if s<0:
#        sp_ph_0 = (1-2*np.mod(l,2))
#        sp_ph_1 = (1-2*np.mod(l-m_arr_1,2))
#        sp_ph_2 = (1-2*np.mod(l+m_arr_1+s,2))
#        sp_ph_3 = (1-2*np.mod(n_arr_1,2))
#
#        sp_ph_4 = np.ones((l+1), dtype=int_prec)
#        sp_ph_5 = (1-2*np.mod(-s-n_arr_1,2))
#        sp_ph_6 = np.ones((l+1), dtype=int_prec)
#        sp_ph_7 = (1-2*np.mod(-s-n_arr_2,2))
#        sp_ph_8 = 1
#
#    else:
#        sp_ph_0 = 1
#        sp_ph_1 = np.ones((l+1), dtype=int_prec)
#        sp_ph_2 = np.ones((l+1), dtype=int_prec)
#        sp_ph_3 = (1-2*np.mod(s+n_arr_1,2))
#
#        sp_ph_4 = (1-2*np.mod(l+m_arr_1,2))
#        sp_ph_5 = (1-2*np.mod(-s-n_arr_2,2))
#        sp_ph_6 = (1-2*np.mod(l-m_arr_1,2))
#
#        sp_ph_7 = (1-2*np.mod(-s-n_arr_2,2))
#        sp_ph_8 = (1-2*np.mod(l,2))

#    sp_ph_l = (1-2*np.mod(l,2))
#    sp_ph_m = (1-2*np.mod(np.arange(-l,0),2))

#    sp_ph_0 = sp_ph_l*sp_ph_m if (s<0) else np.ones((l+1), dtype=int_prec)

    # Positive 'n' indices
    for n in np.arange(0, n_lim + 1):
        ind_ln = l_fa + n
        salmdel_cur = salm_til_arr[ind_ln]

        for m in np.arange(-l, 0):
            #            prod_salmdel = salm_til_arr[ind_ln]* \
            #                int_Del_quad[-m,abs_s]*int_Del_quad[-m,n]

            prod_salmdel = salmdel_cur * \
                int_Del_quad[-m, abs_s] * int_Del_quad[-m, n]

            # Negative 'm' indices
            #sp_ph = (1-2*np.mod(l-m,2)) if (s<0) else 1
            sp_ph = (1 - 2 * ((l - m) % 2)) if (s < 0) else 1
            #sp_ph = sp_ph_l*sp_ph_m[l+m] if (s<0) else 1
            #sp_ph = sp_ph_0[l+m]

            J_mn[m0_int + m, n0_int + n] += sp_ph * prod_salmdel
#            J_mn[m0_int+m, n0_int+n] += sp_ph_0[l+m]*prod_salmdel

            # Positive 'm' indices
            #sp_ph = (1-2*np.mod(l+m+s+n,2)) \
            #    if (s<0) else (1-2*np.mod(s+n,2))

            sp_ph = (1 - 2 * ((l + m + s + n) % 2)) \
                if (s < 0) else (1 - 2 * ((s + n) % 2))

            J_mn[m0_int - m, n0_int + n] += sp_ph * prod_salmdel
            # J_mn[m0_int-m, n0_int+n] += sp_ph_2[l+m]*sp_ph_3[n]* \
            #    prod_salmdel

        # m=0 case
        #sp_ph = (1-2*np.mod(l,2)) if (s<0) else 1
        sp_ph = (1 - 2 * ((l) % 2)) if (s < 0) else 1

        J_mn[m0_int, n0_int + n] += \
            sp_ph * \
            salmdel_cur * \
            int_Del_quad[0, abs_s] * int_Del_quad[0, n]

        # J_mn[m0_int, n0_int+n] += sp_ph_0 * \
        #    salm_til_arr[ind_ln]*int_Del_quad[0,abs_s]*int_Del_quad[0,n]

    # Negative 'n' indices
    for n in np.arange(-n_lim, 0):
        ind_ln = l_fa + n
        salmdel_cur = salm_til_arr[ind_ln]

        for m in np.arange(-l, 0):
            # Negative 'm' indices
            #            prod_salmdel = salm_til_arr[ind_ln]* \
            #                int_Del_quad[-m,abs_s]*int_Del_quad[-m,-n]

            prod_salmdel = salmdel_cur * \
                int_Del_quad[-m, abs_s] * int_Del_quad[-m, -n]

            #sp_ph = 1 if (s<0) else (1-2*np.mod(l+m,2))
            sp_ph = 1 if (s < 0) else (1 - 2 * ((l + m) % 2))

            J_mn[m0_int + m, n0_int + n] += sp_ph * prod_salmdel
            #J_mn[m0_int+m, n0_int+n] += sp_ph_4[l+m]*prod_salmdel

            # Positive 'm' indices
            #sp_ph = (1-2*np.mod(-s-n,2)) \
            #    if (s<0) else (1-2*np.mod(l-s-n-m,2))

            sp_ph = (1 - 2 * ((-s - n) % 2)) \
                if (s < 0) else (1 - 2 * ((l - s - n - m) % 2))

            J_mn[m0_int - m, n0_int + n] += sp_ph * prod_salmdel
            # J_mn[m0_int-m, n0_int+n] += sp_ph_5[n_lim+n]*sp_ph_6[l+m]* \
            #    prod_salmdel

        # m=0 case
        #sp_ph = (1-2*np.mod(-s-n,2)) \
        #    if (s<0) else (1-2*np.mod(l-s-n,2))

        sp_ph = (1 - 2 * ((-s - n) % 2)) \
            if (s < 0) else (1 - 2 * ((l - s - n) % 2))

        J_mn[m0_int, n0_int + n] += \
            sp_ph * salmdel_cur * \
            int_Del_quad[0, abs_s] * \
            int_Del_quad[0, -n]

        # J_mn[m0_int, n0_int+n] += \
        #        sp_ph_7[n_lim+n]*sp_ph_8*salm_til_arr[ind_ln]* \
        #        int_Del_quad[0,abs_s]*int_Del_quad[0,-n]


def par_mask(J_mn_th, l_ran, int_Del_ext, L_th, L_ph,
             s_arr, salm_til_arr, m0, n0, num_fun):
    for l in np.arange(l_ran[0], l_ran[1]):
        # Compute current internal Del values
        int_Del_int = wig.int_Del_Interior(l, int_Del_ext, L_ph)
        int_Del_quad = \
            wig.int_Del_Interior_ExtendQuad(l, int_Del_int, L_ph)

        # Maximal bandlimit in phi restricted by sampling choice
        #n_lim = np.min(np.array([L_ph, l]))
        n_lim = L_ph if (L_ph <= l) else l

        for f in np.arange(0, num_fun):
            if np.abs(s_arr[f]) <= l:
                _int_J_mn_Del_opt(int_prec(l), int_prec(n_lim),
                                  J_mn_th[f], m0, n0, int_Del_quad,
                                  int_prec(s_arr[f]), salm_til_arr[f])

    return J_mn_th


def int_J_mn(s_arr, salm_arr, N_th, N_ph):
    '''
    Compute functional for Fourier transform -
    Reconstruction of integer spin-weight functions from coefficients
    '''
    num_fun = salm_arr.shape[0]

    # Infer band-limits
    L_th = int_prec(N_th / 2 - 2)
    L_ph = int_prec((N_ph - 2) / 2)

    # Infer sampling
    N_th, N_ph = int_prec(2 * (L_th + 2)), int_prec(2 * (L_ph) + 2)

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Compute del face
    int_Del_strip = wig.int_Del_Strip(L_th)
    int_Del_ext = wig.int_Del_Exterior(L_th, int_Del_strip, L_ph)

    # Apply phases
    salm_til_arr = np.copy(salm_arr)

    _int_J_mn_ph(int_prec(s_arr), salm_til_arr, L_th, L_ph)

    # for f in np.arange(0, num_fun):
    #    _int_J_mn_ph_o(int_prec(s_arr[f]), salm_til_arr[f], L_th, L_ph)

    # Allocate space for functional

    J_mn = np.zeros((num_fun, N_th_E, N_ph), dtype=num_prec_complex)
    # Zero indices
    m0, n0 = int_prec(N_th_E / 2), int_prec(N_ph / 2)

    if n_th > 1:
        J_mn_th = \
            np.zeros((n_th, num_fun, N_th_E, N_ph), dtype=num_prec_complex)
        # Partition l range so that intervals scale quadratically
        # This should give a roughly even distribution of workload

        ind_min = np.ceil(np.sqrt((L_th + 1)**2 / n_th * np.arange(0, n_th)))
        ind_max = np.ceil(
            np.sqrt((L_th + 1)**2 / n_th * np.arange(1, n_th + 1)))
        ind_ran = np.zeros(2 * n_th, dtype=int_prec)
        ind_ran[0::2] = int_prec(ind_min)
        ind_ran[1::2] = int_prec(ind_max)
        ind_ran = np.split(ind_ran, n_th)

        l_ran_th = ind_ran

        task_iter = (jl.delayed(par_mask)(J_mn_th[i], l_ran_th[i],
                                          int_Del_ext, L_th, L_ph, s_arr, salm_til_arr, m0, n0, num_fun)
                     for i in np.arange(0, n_th))
        res = par(task_iter)

        J_mn = np.sum(res, axis=0)
    else:

        for l in np.arange(np.min(np.abs(s_arr)), L_th + 1):

            # Compute current internal Del values
            int_Del_int = wig.int_Del_Interior(l, int_Del_ext, L_ph)
            int_Del_quad = \
                wig.int_Del_Interior_ExtendQuad(l, int_Del_int, L_ph)

            # Maximal bandlimit in phi restricted by sampling choice
            #n_lim = np.min(np.array([L_ph, l]))
            n_lim = L_ph if (L_ph <= l) else l

            for f in np.arange(0, num_fun):
                if np.abs(s_arr[f]) <= l:
                    _int_J_mn_Del_opt(int_prec(l), int_prec(n_lim),
                                      J_mn[f], m0, n0, int_Del_quad,
                                      int_prec(s_arr[f]), salm_til_arr[f])

            #_int_J_mn_Del_opt(int_prec(l), int_prec(n_lim), \
            #    J_mn, m0, n0, int_Del_quad, \
            #    int_prec(s_arr), salm_til_arr)

    return J_mn


def _int_J_mn(s_arr, salm_arr, N_th, N_ph):
    '''
    Compute functional for Fourier transform -
    Reconstruction of integer spin-weight functions from coefficients
    '''
    num_fun = salm_arr.shape[0]

    # Infer band-limits
    L_th = int_prec(N_th / 2 - 2)
    L_ph = int_prec((N_ph - 2) / 2)

    # Infer sampling
    N_th, N_ph = int_prec(2 * (L_th + 2)), int_prec(2 * (L_ph) + 2)

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Compute del face
    int_Del_strip = wig.int_Del_Strip(L_th)
    int_Del_ext = wig.int_Del_Exterior(L_th, int_Del_strip, L_ph)

    # Apply phases
    salm_til_arr = np.copy(salm_arr)

    _int_J_mn_ph(int_prec(s_arr), salm_til_arr, L_th, L_ph)

    # for f in np.arange(0, num_fun):
    #    _int_J_mn_ph_o(int_prec(s_arr[f]), salm_til_arr[f], L_th, L_ph)

    # Allocate space for functional

    J_mn = np.zeros((num_fun, N_th_E, N_ph), dtype=num_prec_complex)
    # Zero indices
    m0, n0 = int_prec(N_th_E / 2), int_prec(N_ph / 2)

    for l in np.arange(np.min(np.abs(s_arr)), L_th + 1):

        # Compute current internal Del values
        int_Del_int = wig.int_Del_Interior(l, int_Del_ext, L_ph)
        int_Del_quad = wig.int_Del_Interior_ExtendQuad(l, int_Del_int, L_ph)

        # Maximal bandlimit in phi restricted by sampling choice
        #n_lim = np.min(np.array([L_ph, l]))
        n_lim = L_ph if (L_ph <= l) else l

        for f in np.arange(0, num_fun):
            if np.abs(s_arr[f]) <= l:
                _int_J_mn_Del_opt(int_prec(l), int_prec(n_lim),
                                  J_mn[f], m0, n0, int_Del_quad,
                                  int_prec(s_arr[f]), salm_til_arr[f])

#        _int_J_mn_Del_opt_(int_prec(l), int_prec(n_lim), \
#            J_mn, m0, n0, int_Del_quad, \
#            int_prec(s_arr), salm_til_arr)

    return J_mn


@nu.jit(nopython=True, nogil=True, cache=True)
def _h_int_K_mn_ph(h_s2_arr, h_salm_arr, h_L2_th, h_L2_ph):
    '''
    Apply phase factors to salm coefficients
    '''
    num_fun = h_salm_arr.shape[0]

    for l2 in np.arange(int_prec(np.min(np.abs(h_s2_arr))), h_L2_th + 1, 2):
        l_sqFa = np.sqrt((l2 + 1) / (4 * np.pi))

        n_lim = h_L2_ph if (h_L2_ph <= l2) else l2

        for n2 in np.arange(-n_lim, n_lim + 1, 2):
            # Index for salm
            h_ind_ln = int_prec(1 / 2 * (n2 + l2 * (l2 / 2 + 1) - 1 / 2))

            for f in np.arange(0, num_fun):
                ph = (1j)**(h_s2_arr[f] / 2 - n2 / 2)
                h_salm_arr[f][h_ind_ln] = \
                    ph * l_sqFa * h_salm_arr[f][h_ind_ln]


@nu.jit(nopython=True, nogil=True, cache=True)
def _h_int_K_mn_Del(l2, n_lim, K_mn, m1, n1, h_int_Del_quad,
                    h_s2_arr, h_s_arr, h_salm_til_arr):
    '''
    Apply Del symmetries and compute K_mn functionals
    '''

    num_fun = h_salm_til_arr.shape[0]

    # Positive 'n' indices
    for n2 in np.arange(1, n_lim + 1, 2):

        # Index for salm
        h_ind_ln = int_prec(1 / 2 * (n2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in np.arange(0, num_fun):
            if np.abs(h_s2_arr[f]) <= l2:

                # Take care of negative m
                for m2 in np.arange(-l2, 1, 2):
                    # Negative 'm' indices
                    sp_ph = (-1)**(l2 / 2 + m2 / 2) if (h_s_arr[f] < 0) else 1

                    # Array indices
                    m_i, n_i = int_prec((np.abs(m2) - 1) / 2), \
                        int_prec((n2 - 1) / 2)
                    s_i = int_prec((np.abs(h_s2_arr[f]) - 1) / 2)

                    K_mn[f, int_prec(m1 + m2), int_prec(n1 + n2)] += \
                        sp_ph * \
                        h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]

                    # Positive 'm' indices
                    sp_ph = (-1)**(l2 / 2 - m2 / 2) if (h_s_arr[f] < 0) else 1

                    K_mn[f, int_prec(m1 - m2), int_prec(n1 + n2)] += \
                        -(-1)**(np.abs(h_s_arr[f]) + n2 / 2) * \
                        sp_ph * \
                        h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]

    # Negative 'n' indices
    for n2 in np.arange(-n_lim, 1, 2):

        # Index for salm
        h_ind_ln = int_prec(1 / 2 * (n2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in np.arange(0, num_fun):
            if np.abs(h_s2_arr[f]) <= l2:

                # Take care of negative m
                for m2 in np.arange(-l2, 1, 2):

                    # Negative 'm' indices
                    sp_ph = (-1)**(l2 / 2 + m2 / 2) if (h_s_arr[f] < 0) else 1

                    # Array indices
                    m_i, n_i = int_prec((np.abs(m2) - 1) / 2), \
                        int_prec((np.abs(n2) - 1) / 2)
                    s_i = int_prec((np.abs(h_s2_arr[f]) - 1) / 2)

                    K_mn[f, int_prec(m1 + m2), int_prec(n1 + n2)] += \
                        sp_ph * \
                        (-1)**(l2 / 2 + m2 / 2) * \
                        h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]

                    # Positive 'm' indices
                    sp_ph = (-1)**(l2 / 2 - m2 / 2) if (h_s_arr[f] < 0) else 1

                    K_mn[f, int_prec(m1 - m2), int_prec(n1 + n2)] += \
                        (-1)**(l2 / 2 + np.abs(h_s_arr[f]) + n2 / 2 - m2 / 2) * \
                        sp_ph * \
                        h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]


@nu.jit(nopython=True, nogil=True, cache=True)
def _h_int_K_mn_Del_opt(l2, n_lim, K_mn, m1, n1, h_int_Del_quad,
                        h_s2_arr, h_s_arr, h_salm_til_arr):
    '''
    Apply Del symmetries and compute K_mn functionals
    '''

    num_fun = h_salm_til_arr.shape[0]

    abs_h_s2_arr = np.abs(h_s2_arr)
    abs_h_s_arr = np.abs(h_s_arr)

    # Positive 'n' indices
    for n2 in np.arange(1, n_lim + 1, 2):

        # Index for salm
        h_ind_ln = int_prec(1 / 2 * (n2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in np.arange(0, num_fun):
            if np.abs(h_s2_arr[f]) <= l2:

                # Take care of negative m
                for m2 in np.arange(-l2, 1, 2):
                    # Array indices
                    m_i, n_i = int_prec((np.abs(m2) - 1) / 2), \
                        int_prec((n2 - 1) / 2)
                    s_i = int_prec((abs_h_s2_arr[f] - 1) / 2)

                    # Array-del product
                    h_salm_del_prod = h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]

                    # Negative 'm' indices
                    sp_ph = (1 - 2 * np.mod(l2 / 2 + m2 / 2, 2)
                             ) if (h_s_arr[f] < 0) else 1

                    K_mn[f, m1 + m2, n1 + n2] += sp_ph * h_salm_del_prod

                    # Positive 'm' indices
                    sp_ph = (1 - 2 * np.mod(l2 / 2 - m2 / 2 + abs_h_s_arr[f] + n2 / 2, 2)) \
                        if (h_s_arr[f] < 0) else \
                        (1 - 2 * np.mod(abs_h_s_arr[f] + n2 / 2, 2))

                    K_mn[f, m1 - m2, n1 + n2] += -sp_ph * h_salm_del_prod

    # Negative 'n' indices
    for n2 in np.arange(-n_lim, 1, 2):

        # Index for salm
        h_ind_ln = int_prec(1 / 2 * (n2 + l2 * (l2 / 2 + 1) - 1 / 2))

        for f in np.arange(0, num_fun):
            if np.abs(h_s2_arr[f]) <= l2:
                # Take care of negative m
                for m2 in np.arange(-l2, 1, 2):
                    # Array indices
                    m_i, n_i = int_prec((np.abs(m2) - 1) / 2), \
                        int_prec((np.abs(n2) - 1) / 2)
                    s_i = int_prec((abs_h_s2_arr[f] - 1) / 2)

                    # Array-del product
                    h_salm_del_prod = h_salm_til_arr[f][h_ind_ln] * \
                        h_int_Del_quad[m_i, s_i] * \
                        h_int_Del_quad[m_i, n_i]

                    # Negative 'm' indices
                    sp_ph = 1 if (h_s_arr[f] < 0) else (
                        1 - 2 * np.mod(l2 / 2 + m2 / 2, 2))

                    K_mn[f, m1 + m2, n1 + n2] += sp_ph * h_salm_del_prod

                    # Positive 'm' indices
                    sp_ph = (1 - 2 * np.mod(l2 + abs_h_s_arr[f] + n2 / 2 - m2, 2)) if \
                        (h_s_arr[f] < 0) else \
                        (1 - 2 * np.mod(l2 / 2 +
                                        abs_h_s_arr[f] + n2 / 2 - m2 / 2, 2))

                    K_mn[f, m1 - m2, n1 + n2] += sp_ph * h_salm_del_prod


def h_int_K_mn(h_s_arr, h_salm_arr, h_N_th, h_N_ph):
    '''
    Compute functional for Fourier transform -
    Reconstruction of half integer spin-weight functions from coefficients
    '''
    num_fun = h_salm_arr.shape[0]

    # Integer spin-weight representation (double val.)
    h_s2_arr = int_prec(2 * h_s_arr)

    # Infer band-limits (retain double)
    h_L2_th = h_N_th - 1
    h_L2_ph = h_N_ph - 1

    # Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2 * (h_N_th - 1), 4 * (h_N_th - 1)
    h_N_ph_E = 2 * h_N_ph

    # Compute del face
    h_int_Del_strip = wig.h_int_Del_Strip(h_L2_th)
    h_int_Del_ext = wig.h_int_Del_Exterior(h_L2_th, h_int_Del_strip, h_L2_ph)

    # Apply phases
    h_salm_til_arr = np.copy(h_salm_arr)
    # Ensure we pass correct types
    _h_int_K_mn_ph(h_s2_arr, h_salm_til_arr, h_L2_th, h_L2_ph)

    # Allocate space for functional

    K_mn = np.zeros((num_fun, h_N_th_EE, h_N_ph_E), dtype=num_prec_complex)

    # 1/2 indices
    m1, n1 = int_prec(h_N_th_E), int_prec(h_N_ph)

    for l2 in np.arange(np.min(np.abs(h_s2_arr)), h_L2_th + 1, 2):
        # Compute current internal Del values
        h_int_Del_int = wig.h_int_Del_Interior(l2, h_int_Del_ext, h_L2_ph)
        h_int_Del_quad = \
            wig.h_int_Del_Interior_ExtendQuad(l2, h_int_Del_int, h_L2_ph)

        # Maximal bandlimit in phi restricted by sampling choice
        n_lim = h_L2_ph if (h_L2_ph <= l2) else l2

        _h_int_K_mn_Del_opt(l2, n_lim, K_mn, m1, n1, h_int_Del_quad,
                            h_s2_arr, h_s_arr, h_salm_til_arr)

    return K_mn


def int_sf(s_arr, salm_arr, N_th, N_ph):
    '''
    Compute integer spin-weighted functions on S2 from salm
    '''

    num_fun = salm_arr.shape[0]
    #salm_arr_cp = np.copy(salm_arr)

    # Compute J_mn functional
    J_mn_arr = int_J_mn(s_arr, salm_arr, N_th, N_ph)

    # Allocate space for functions
    sf_arr = np.zeros((num_fun, N_th, N_ph), dtype=num_prec_complex)

    for f in np.arange(0, num_fun):
        sf_tmp = FSi.ifftshift(FSi.ifft(FSi.fftshift(J_mn_arr[f]),
                                        normalized=True))[:N_th, :]
        sf_arr[f, :, :] = np.flipud(sf_tmp)

    return sf_arr


def h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph):
    '''
    Compute half integer spin-weighted functions on S2 from salm
    '''

    num_fun = h_salm_arr.shape[0]
    #h_salm_arr_cp = np.copy(h_salm_arr)

    # Compute K_mn functional
    K_mn_arr = h_int_K_mn(h_s_arr, h_salm_arr, h_N_th, h_N_ph)

    # Allocate space for functions
    h_sf_arr = np.zeros((num_fun, h_N_th, h_N_ph), dtype=num_prec_complex)

    for f in np.arange(0, num_fun):
        h_sf_arr[f, :, :] = FSi.ifftshift(FSi.ifft(FSi.fftshift(K_mn_arr[f]),
                                                   normalized=True))[:h_N_th, :h_N_ph]
    return h_sf_arr


if __name__ == '__main__':
    # Instantiate Fourier object
    FSi = FS.FourierSeries()

    ########
    # Integer
    ########

    # Num. nodes on S^2 grid [must be even]
    # To sample up to l=L /\ m=L take: N_th=2(L+2) & N_ph=2L+2
    L_th, L_ph = 128, 128
    N_th, N_ph = 2 * (L_th + 2), 2 * (L_ph) + 2

    # Extension in theta direction requires more points
    N_th_E = 2 * (N_th - 1)

    # Extended domain lattice
    int_th = np.pi / (N_th - 1) * np.arange(0, (N_th_E))
    int_ph = 2 * np.pi / N_ph * np.arange(0, (N_ph - 1) + 1)

    # Spin-weights
    s_arr = np.array([0, 2, -1])

    # Test functions [build from analytical defn. of Wigner D]
    int_fun_a = lambda th, ph: swsh_an.sylm(s_arr[0], 1, -1, th, ph) + \
        swsh_an.sylm(s_arr[0], 2, -2, th, ph) + \
        swsh_an.sylm(s_arr[0], 4, 2, th, ph)
    int_fun_b = lambda th, ph: swsh_an.sylm(s_arr[1], 2, 2, th, ph) + \
        swsh_an.sylm(s_arr[1], 2, -1, th, ph) + \
        swsh_an.sylm(s_arr[1], 3, 3, th, ph)

    int_fun_c = lambda th, ph: swsh_an.sylm(s_arr[2], 2, -1, th, ph) + \
        11 / 10 * swsh_an.sylm(s_arr[2], 4, -4, th, ph) - \
        22 / 10 * swsh_an.sylm(s_arr[2], 4, 4, th, ph)

    # Evaluate function on 2-sphere
    int_fun_a_N_S2 = int_fun_a(int_th[:N_th], int_ph[:N_ph])
    int_fun_b_N_S2 = int_fun_b(int_th[:N_th], int_ph[:N_ph])
    int_fun_c_N_S2 = int_fun_c(int_th[:N_th], int_ph[:N_ph])

    # Coefficient indexing map:
    ind_map = lambda li, mi: li * (li + 1) + mi

    sz_salm = (L_th + 1)**2
    num_fun = 3
    salm_arr = np.zeros((num_fun, sz_salm), dtype=num_prec_complex)

    # Populate test coefficients
    # First function
    salm_arr[0][ind_map(1, -1)] = 1
    salm_arr[0][ind_map(2, -2)] = 1
    salm_arr[0][ind_map(4, 2)] = 1

    # Second function
    salm_arr[1][ind_map(2, -1)] = 1
    salm_arr[1][ind_map(2, 2)] = 1
    salm_arr[1][ind_map(3, 3)] = 1

    salm_arr[2][ind_map(2, -1)] = 1
    salm_arr[2][ind_map(4, -4)] = 11 / 10
    salm_arr[2][ind_map(4, 4)] = -22 / 10

    #J_mn = int_J_mn(s_arr, salm_arr, N_th, N_ph)

    sf = int_sf(s_arr, salm_arr, N_th, N_ph)

#
# :D
#
