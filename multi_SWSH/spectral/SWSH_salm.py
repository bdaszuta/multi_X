#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Tue Dec 29 16:52:16 2015

@author: Boris Daszuta
@function: Computation of coefficients

"""

import spectral.SWSH_Imn as SWSHsImn

import special_functions.wigner_d_TN as wig

import numpy as np
import numba as nu

num_prec_complex = np.complex128
uint_prec = np.uint64
int_prec = np.int64

@nu.autojit(nopython=True, nogil=True, cache=True)
def _int_salm(l, l_sqFa, n_ma, m_ma, s_arr, K_mn, int_Del_quad, salm_arr):
    '''
    Jit compilation of loop for Del calculation
    '''
    num_fun = salm_arr.shape[0]
    
    #Loop limits
    m_lim = np.zeros(2, dtype=int_prec)
    m_lim[0], m_lim[1] = n_ma, l
    m_lim = np.min(m_lim)+1

    q_lim = np.zeros(2, dtype=int_prec)
    q_lim[0], q_lim[1] = m_ma, l
    q_lim = np.min(q_lim) + 1
    
    #ph_s = (1j)**(s_arr)
    #ph_m = (1j)**(-np.arange(0, m_lim))    
    
    #Take care of positive m
    for m in np.arange(0, m_lim):
        #Index for salm
        ind_lm = l*(l+1)+m
        
        for f in np.arange(0, num_fun):
            if np.abs(s_arr[f])<=l:
                for q in np.arange(0, q_lim):
                    
                    #Use symmetries to treat negative spin-weight terms
                    if s_arr[f]<0:
                        #sp_ph = (-1)**(l-q)
                        sp_ph = (1-2*np.mod(l-q,2))
                    else:
                        sp_ph = 1
                    
                    del_sp = sp_ph*int_Del_quad[q,np.abs(s_arr[f])]
                    
                    salm_arr[f,ind_lm] += \
                        int_Del_quad[q,m]*K_mn[f,q,n_ma+m]*del_sp
                
                #Apply scaling and phase
                salm_arr[f,ind_lm] = \
                    salm_arr[f,ind_lm]*l_sqFa*(1j**(s_arr[f]-m))

                #salm_arr[f,ind_lm] = \
                #    salm_arr[f,ind_lm]*l_sqFa*(ph_s[f]*1j**(-m))



    #Take care of negative m
    for m in np.arange(-m_lim+1, 0):
        #Index for salm
        ind_lm = l*(l+1)+m
        
        for f in np.arange(0, num_fun):
            if np.abs(s_arr[f])<=l:
                for q in np.arange(0, q_lim):
                    
                    #Use symmetries to treat negative spin-weight terms
                    if s_arr[f]<0:
                        #sp_ph = (-1)**(l-q)
                        sp_ph = (1-2*np.mod(l-q,2))
                    else:
                        sp_ph = 1
                    
                    del_sp = sp_ph*int_Del_quad[q,np.abs(s_arr[f])]
                    
                    #salm_arr[f,ind_lm] += \
                    #    (-1)**(l-q)*int_Del_quad[q,-m]* \
                    #    K_mn[f,q,n_ma+m]*del_sp
                    salm_arr[f,ind_lm] += \
                        (1-2*np.mod(l-q,2))*int_Del_quad[q,-m]* \
                        K_mn[f,q,n_ma+m]*del_sp


    
                #Apply scaling and phase
                salm_arr[f,ind_lm] = \
                    salm_arr[f,ind_lm]*l_sqFa*(1j**(s_arr[f]-m))

def int_salm(s_arr, f_arr):
    '''
    Arrays are expected as input (see interface)
    '''

    #[If leaks occur] Prevent memory issues when called from external library
    #s_arr = np.copy(s_arr)
    #f_arr = np.copy(f_arr)

    #Get total number of input functions
    num_fun = (f_arr.shape)[0]
    
    #Infer extension sampling
    N_th = f_arr[0].shape[0]
    N_th_E = 2*(N_th-1)
    N_ph = f_arr[0].shape[1]
    L_ph = int_prec((N_ph-2)/2)

    #Computation of weights
    n_wei = SWSHsImn.int_wei_th(N_th_E)

    #Calculate I_mn for each function
    I_mn_arr = np.zeros((num_fun, N_th_E-1, N_ph-1), dtype=num_prec_complex)
    for f in np.arange(num_fun):
        I_mn_arr[f] = \
            SWSHsImn.int_quadrature(s_arr[f], f_arr[f], n_wei)[1:,1:]
    
    #Infer I_mn dimensions and hence maximal m,n values
    m_ma, n_ma = int_prec((N_th_E-2)/2), int_prec((N_ph-2)/2)

    #Prepare Kmn
    K_mn = I_mn_arr[:,m_ma:,:]
    
    for f in np.arange(num_fun):
        #Apply symmetry
        #I_mn_ph = np.array([(-1)**(np.arange(-n_ma, n_ma+1)+s_arr[f])])
        I_mn_ph = \
            np.array([(1-2*np.mod(np.arange(-n_ma, n_ma+1)+s_arr[f],2))])

        K_mn[f, 1:,:] += \
            np.repeat(I_mn_ph, m_ma, axis=0)*np.flipud(I_mn_arr[f,:m_ma,:])


    #Infer maximal required Del element
    L = int_prec((N_th_E-6)/4)
        
    #Assign space for coefficients
    salm_arr = np.zeros((num_fun, (L+1)**2), dtype=num_prec_complex)


    #Compute del face
    int_Del_strip = wig.int_Del_Strip(L)
    int_Del_ext = wig.int_Del_Exterior(L, int_Del_strip, L_ph)

    for l in np.arange(np.min(np.abs(s_arr)), L+1):
        l_sqFa = np.sqrt((2*l+1)/(4*np.pi))

        #Compute current internal Del values
        int_Del_int = wig.int_Del_Interior(l, int_Del_ext, L_ph)
        int_Del_quad = wig.int_Del_Interior_ExtendQuad(l, int_Del_int, L_ph)
        
        
        _int_salm(l, l_sqFa, n_ma, m_ma, \
            int_prec(s_arr), K_mn, int_Del_quad, salm_arr)


    return salm_arr


@nu.autojit(nopython=True, nogil=True, cache=True)
def _h_int_salm(l2, l_sqFa, n_ma, m_ma, I_mn_m_ind, I_mn_n_ind, s_arr, J_mn, \
    h_int_Del_quad, salm_arr):
    '''
    Jit compilation of loop for Del calculation
    '''
    num_fun = salm_arr.shape[0]

    #Loop limits
    m_lim = np.zeros(2, dtype=int_prec)
    m_lim[0], m_lim[1] = n_ma, l2
    #Increase for correct arange maxima
    m_lim = np.min(m_lim)+1

    q_lim = np.zeros(2, dtype=int_prec)
    q_lim[0], q_lim[1] = m_ma, l2
    #Increase for correct arange maxima
    q_lim = np.min(q_lim)+1

    #Take care of positive m
    for m2 in np.arange(1, m_lim, 2):
        #Index for salm
        ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))

        
        for f in np.arange(0, num_fun):
            if int_prec(np.abs(2*s_arr[f]))<=l2:
                for q2 in np.arange(1, q_lim, 2):
                    #Use symmetries to treat negative spin-weight terms
                    if s_arr[f]<0:
                        #sp_ph = (-1)**((l2-q2)/2)
                        sp_ph = (1-2*np.mod((l2-q2)/2,2))
                    else:
                        sp_ph = 1
                    
                    #Array indices
                    q_i = int_prec((q2-1)/2)
                    s_i_f = int_prec((np.abs(s_arr[f]*2)-1)/2)
                    m_i = int_prec((m2-1)/2)
                    
                    del_sp = sp_ph*h_int_Del_quad[q_i, s_i_f]
                    
                    salm_arr[f,ind_lm] += \
                        h_int_Del_quad[q_i,m_i]* \
                        J_mn[f,q_i,int_prec(m_i+I_mn_n_ind)]*del_sp
                
                #Apply scaling and phase
                salm_arr[f,ind_lm] = \
                    salm_arr[f,ind_lm]*l_sqFa*(1j**(s_arr[f]-m2/2))

    #Take care of negative m
    for m2 in np.arange(-m_lim+1, 1, 2):
        #Index for salm
        ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
        
        for f in np.arange(0, num_fun):
            if int_prec(np.abs(2*s_arr[f]))<=l2:
                for q2 in np.arange(1, q_lim, 2):
                    #Use symmetries to treat negative spin-weight terms
                    if s_arr[f]<0:
                        #sp_ph = (-1)**((l2-q2)/2)
                        sp_ph = (1-2*np.mod((l2-q2)/2,2))
                    else:
                        sp_ph = 1
                    

                    #Array indices
                    q_i = int_prec((q2-1)/2)
                    s_i_f = int_prec((np.abs(s_arr[f]*2)-1)/2)
                    m_i = int_prec((np.abs(m2)-1)/2)

                    del_sp = sp_ph*h_int_Del_quad[q_i,s_i_f]

                    #salm_arr[f,ind_lm] += \
                    #    (-1)**((l2-q2)/2)*h_int_Del_quad[q_i,m_i]* \
                    #    J_mn[f,q_i,int_prec(I_mn_n_ind-m_i-1)]*del_sp

                    salm_arr[f,ind_lm] += \
                        (1-2*np.mod((l2-q2)/2,2))*h_int_Del_quad[q_i,m_i]* \
                        J_mn[f,q_i,int_prec(I_mn_n_ind-m_i-1)]*del_sp
    
                #Apply scaling and phase
                salm_arr[f,ind_lm] = \
                    salm_arr[f,ind_lm]*l_sqFa*(1j**(s_arr[f]+np.abs(m2)/2))


def h_int_salm(s_arr, f_arr):
    '''
    Arrays are expected as input (see interface)
    '''
    
    #Get total number of input functions
    num_fun = (f_arr.shape)[0]
    
    #Infer extension sampling
    h_N_th = f_arr[0].shape[0]
    h_N_ph = f_arr[1].shape[1]
    h_L_ph = int_prec((h_N_ph-1))
    
    #Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2*(h_N_th-1), 4*(h_N_th-1)
    h_N_ph_E = 2*h_N_ph
    
    #Computation of weights
    h_n_wei_th, h_n_wei_ph = SWSHsImn.h_int_wei(h_N_th_EE, h_N_ph_E)



    dImn_m = int_prec((h_N_th_E-2 - h_N_th)/2)
    

    #Calculate I_mn for each function
    I_mn_arr = np.zeros((num_fun, h_N_th, h_N_ph), \
        dtype=num_prec_complex)
    for f in np.arange(num_fun):
        I_mn_arr[f] = \
            SWSHsImn.h_int_quadrature(s_arr[f], f_arr[f], \
                h_n_wei_th, h_n_wei_ph)[dImn_m:-dImn_m,:]

    #Infer I_mn dimensions and hence maximal m,n values
    #[Integral part to be divided by 2]
    m_ma, n_ma = h_N_th-1, h_N_ph-1

    
    #Index for m=1/2
    I_mn_m_ind = h_N_th/2
    I_mn_n_ind = h_N_ph/2
    

    #Prepare Jmn
    J_mn = I_mn_arr[:,I_mn_m_ind:,:]
    
    
    for f in np.arange(num_fun):
        #Compute phase
        pwr_v = int_prec(np.arange(-n_ma, n_ma+1,2)/2+s_arr[f])
        I_mn_ph = -np.array([(-1)**pwr_v])

        I_mn_ph = np.repeat(I_mn_ph, I_mn_m_ind, axis=0)

        J_mn[f,:,:] += \
            I_mn_ph*np.flipud(I_mn_arr[f,:I_mn_m_ind,:])

    #return J_mn

    #m_ma also functions as the maximal required Del element
    L2 = m_ma

    #Assign space for coefficients
    sz_salm = int_prec(3/4+(L2/2)*(2+L2/2))
    salm_arr = np.zeros((num_fun, sz_salm), dtype=num_prec_complex)

    #Compute del face
    h_int_Del_strip = wig.h_int_Del_Strip(L2)
    h_int_Del_ext = wig.h_int_Del_Exterior(L2, h_int_Del_strip, h_L_ph)

    
    for l2 in np.arange(np.min(np.abs(2*s_arr)), L2+1, 2):
        l_sqFa = np.sqrt((l2+1)/(4*np.pi))
        
        #Compute current internal Del values
        h_int_Del_int = wig.h_int_Del_Interior(l2, h_int_Del_ext, h_L_ph)
        h_int_Del_quad = \
            wig.h_int_Del_Interior_ExtendQuad(l2, h_int_Del_int, h_L_ph)

        _h_int_salm(l2, l_sqFa, n_ma, m_ma, I_mn_m_ind, I_mn_n_ind, s_arr, \
            J_mn, h_int_Del_quad, salm_arr)
    
    return salm_arr

if __name__ == '__main__':
    pass

#
# :D
#
