#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Mon Dec 21 19:04:29 2015

@author: Boris Daszuta
@function: Interface for construction of SWSH transformations.

"""

import spectral.Interface_FourierSeries as FS
import spectral.SWSH_Imn as SWSHsImn
import spectral.SWSH_salm as SWsalm
import spectral.SWSH_bwd as SWbwd

import numpy as np
import numba as nu

#Test analytically defined sYlm
import special_functions.swsh_AnSum as swsh_an

num_prec_complex = np.complex128
uint_prec = np.uint64
int_prec = np.int64

#Fourier object instance
FSi = None

#Boolean for caching reduced Wigner elements
cache_del = False


def int_salm(s_arr, f_arr):
    '''
    Compute salm from integer spin-weighted functions sampled on S2
    '''
    
    #Ensure we deal with arrays    
    if np.isscalar(s_arr):
        s_arr = np.array([s_arr],dtype=int_prec)
    if len(f_arr.shape) == 2:
        f_arr = np.array([f_arr],dtype=num_prec_complex)

    return SWsalm.int_salm(s_arr, f_arr)

def h_int_salm(s_arr, f_arr):
    '''
    Compute salm from half-integer spin-weighted functions sampled on S2
    '''
    
    #Ensure we deal with arrays    
    if np.isscalar(s_arr):
        s_arr = np.array([s_arr],dtype=np.float64)
    if len(f_arr.shape) == 2:
        f_arr = np.array([f_arr],dtype=num_prec_complex)

    return SWsalm.h_int_salm(s_arr, f_arr)

def int_sf(s_arr, salm_arr_i, N_th, N_ph):
    '''
    Compute integer spin-weighted functions on S2 from salm
    '''
    salm_arr = np.copy(salm_arr_i)
    #Ensure we deal with arrays    
    if np.isscalar(s_arr):
        s_arr = np.array([s_arr],dtype=np.float64)
    if len(salm_arr.shape) == 1:
        salm_arr = np.array([salm_arr],dtype=num_prec_complex)
    
    return SWbwd.int_sf(s_arr, salm_arr, N_th, N_ph)

def h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph):
    '''
    Compute integer spin-weighted functions on S2 from salm
    '''

    #Ensure we deal with arrays    
    if np.isscalar(h_s_arr):
        h_s_arr = np.array([h_s_arr],dtype=np.float64)
    if len(h_salm_arr.shape) == 1:
        h_salm_arr = np.array([h_salm_arr],dtype=num_prec_complex)
    
    return SWbwd.h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph)


def int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    '''
    Generate test coefficients with (uniform-random) entries in:
    re(salm)\in[-1,1] /\ im(salm)\in[-1,1]
    
    >s_arr ~ Spin-weights
    >L_r_th ~ Maximal L_th to populate
    >L_r_ph ~ Maximal L_ph to populate
    >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    '''
    num_fun = s_arr.shape[0]

    #Begin by dense construction:
    sz_salm = (L_th_tr+1)**2
    
    #salm = np.zeros((num_fun, sz_salm), dtype=num_prec_complex)

    salm = (1-2*np.random.rand(num_fun, sz_salm)) + \
        1j*(1-2*np.random.rand(num_fun, sz_salm))

    @nu.autojit(nopython=True, nogil=True, cache=True)
    def _strip_tr(num_fun_i, s_arr_i, L_r_th_i, L_r_ph_i, L_th_tr, salm_i):
        '''
        Strip values that should be zero based on truncation
        '''
        for f in np.arange(0, num_fun_i):
            #l<|s| should be 0
            for l in np.arange(0, np.abs(s_arr_i[f])):
                for m in np.arange(-l,l+1):
                    #Index for salm
                    ind_lm = l*(l+1)+m
                    salm_i[f, ind_lm] = 0
            
            #m>L_r_ph should be 0
            for l in np.arange(np.abs(s_arr_i[f]), L_r_th_i+1):
                for m in np.arange(-l, -L_r_ph_i):
                    #Index for salm
                    ind_lm = l*(l+1)+m
                    salm_i[f, ind_lm] = 0

                for m in np.arange(L_r_ph_i+1, l+1):
                    #Index for salm
                    ind_lm = l*(l+1)+m
                    salm_i[f, ind_lm] = 0

            #l>L_r_th should be 0
            for l in np.arange(L_r_th_i+1, L_th_tr+1):
                for m in np.arange(-l, l+1):
                    #Index for salm
                    ind_lm = l*(l+1)+m
                    salm_i[f, ind_lm] = 0

    _strip_tr(num_fun, s_arr, L_r_th, L_r_ph, L_th_tr, salm)
    
    return salm


def h_int_gen_rand_salm(s_arr, L_r_th, L_r_ph, L_th_tr):
    '''
    Generate test coefficients with (uniform-random) entries in:
    re(salm)\in[-1,1] /\ im(salm)\in[-1,1]
    
    >s_arr ~ Spin-weights
    >L_r_th ~ Maximal L_th to populate
    >L_r_ph ~ Maximal L_ph to populate
    >L_th_tr ~ Bandlimit (use for zero-padding to a size)
    '''
    num_fun = s_arr.shape[0]

    L2_r_th, L2_r_ph, L2_th_tr = 2*L_r_th, 2*L_r_ph, 2*L_th_tr    

    #Begin by dense construction:
    h_sz_salm = int_prec(3/4+(L2_th_tr/2)*(2+L2_th_tr/2))
    
    #salm = np.zeros((num_fun, sz_salm), dtype=num_prec_complex)

    h_salm = (1-2*np.random.rand(num_fun, h_sz_salm)) + \
        1j*(1-2*np.random.rand(num_fun, h_sz_salm))

    @nu.autojit(nopython=True, nogil=True, cache=True)
    def _strip_tr(num_fun_i, s_arr_i, \
        L2_r_th_i, L2_r_ph_i, L2_th_tr, h_salm_i):
        '''
        Strip values that should be zero based on truncation
        '''
        for f in np.arange(0, num_fun_i):
            #l<|s| should be 0
            for l2 in np.arange(1, int_prec(np.abs(2*s_arr_i[f])), 2):
                for m2 in np.arange(-l2,l2+1, 2):
                    #Index for salm
                    ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
                    h_salm_i[f, ind_lm] = 0


            #m>L_r_ph should be 0
            for l2 in np.arange(int_prec(np.abs(2*s_arr_i[f])), \
                L2_r_th_i+1, 2):
                for m2 in np.arange(-l2, -L2_r_ph_i, 2):
                    #Index for salm
                    ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
                    h_salm_i[f, ind_lm] = 0

                for m2 in np.arange(L2_r_ph_i+2, l2+1, 2):
                    #Index for salm
                    ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
                    h_salm_i[f, ind_lm] = 0

                    
            #l>L_r_th should be 0
            for l2 in np.arange(L2_r_th_i+2, L2_th_tr+1, 2):
                for m2 in np.arange(-l2, l2+1, 2):
                    #Index for salm
                    ind_lm = int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
                    h_salm_i[f, ind_lm] = 0
                    

    _strip_tr(num_fun, s_arr, L2_r_th, L2_r_ph, L2_th_tr, h_salm)
    
    return h_salm

def xform_prototype():
    '''
    Test some transformations
    '''

    #Instantiate Fourier object
    FSi = FS.FourierSeries()
    SWSHsImn.FSi = FSi
    SWbwd.FSi = FSi
    
    ########
    #Integer
    ########

    
    #Num. nodes on S^2 grid [must be even]
    #To sample up to l=L /\ m=L take: N_th=2(L+2) & N_ph=2L+2
    L_th, L_ph = 128, 2
    N_th, N_ph = 2*(L_th+2), 2*L_ph+2

    #Extension in theta direction requires more points
    N_th_E = 2*(N_th-1)

    #Extended domain lattice
    int_th = np.pi/(N_th-1)*np.arange(0, (N_th_E))
    int_ph = 2*np.pi/N_ph*np.arange(0, (N_ph-1)+1)


    #Spin-weights
    s_arr = np.array([-1, 2])

    #Test functions [build from analytical defn. of Wigner D]
    int_fun_a = lambda th, ph: swsh_an.sylm(s_arr[0],2,1,th,ph) + \
        11/10*swsh_an.sylm(s_arr[0],4,-4,th,ph) - \
        22/10*swsh_an.sylm(s_arr[0],4,4,th,ph)

    int_fun_b = lambda th, ph: swsh_an.sylm(s_arr[1],2,-2,th,ph) + \
        -5/4*swsh_an.sylm(s_arr[1],3,0,th,ph) + \
        -2*swsh_an.sylm(s_arr[1],4,4,th,ph)


    #Evaluate function on 2-sphere
    int_fun_a_N_S2 = int_fun_a(int_th[:N_th], int_ph[:N_ph])
    int_fun_b_N_S2 = int_fun_b(int_th[:N_th], int_ph[:N_ph])
    

    #Indexing map:
    ind_map = lambda li,mi: li*(li+1)+mi
    
    print('=>integer: fwd')
    #Compute coefficients
    sf_arr = np.array([int_fun_a_N_S2, int_fun_b_N_S2])
    salm_arr = int_salm(s_arr, sf_arr)

    #Throw small values
    #salm_arr[np.abs(salm_arr)<1e-12] = 0

    #Inspect a particular coefficient value
    #print(salm_arr[0][ind_map(4,-4)])

    print('=>integer: bwd')
    sf_rec_arr = int_sf(s_arr, salm_arr, N_th, N_ph)
    
    #Do comparison
    num_fun = salm_arr.shape[0]
    print('>Error inspection')
    for f in np.arange(0, num_fun):
        print('sf->salm->sf_til Err @ fun=' + str(f) +' ' + \
            str(np.max(np.abs(sf_arr[f]- sf_rec_arr[f]))))


    print('=>integer: (rand fwd/bwd pairs)')
    s_r_arr = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])

    salm_r_arr = int_gen_rand_salm(s_r_arr, L_th, L_ph, L_th)
    sf_r_rec_arr = int_sf(s_r_arr, salm_r_arr, N_th, N_ph)
    salm_r_rec_arr = int_salm(s_r_arr, sf_r_rec_arr)

    num_fun_r = salm_r_arr.shape[0]

    print('>Error inspection')
    for f in np.arange(0, num_fun_r):
        print('s:' + str(s_r_arr[f]) + ' salm->sf->salm_til Err @ ' + \
            str(np.max(np.abs(salm_r_arr[f] - salm_r_rec_arr[f]))))
    


    #############
    #Half-integer
    #############

    #Spin-weights
    h_s_arr = np.array([1/2, 3/2])

    #Test functions [build from analytical defn. of Wigner D]
    h_int_fun_a = lambda th, ph: swsh_an.sylm(h_s_arr[0],1/2,-1/2,th,ph) + \
        11/10*swsh_an.sylm(h_s_arr[0],3/2,1/2,th,ph)+ \
        0.1*swsh_an.sylm(h_s_arr[0],11/2,7/2,th,ph)

    h_int_fun_b = lambda th, ph: swsh_an.sylm(h_s_arr[1],3/2,-1/2,th,ph) + \
        -5/4*swsh_an.sylm(h_s_arr[1],11/2,11/2,th,ph)


    #Num. nodes on S^2 grid [must be even]
    #To sample up to l=L /\ m=L take: h_N_th=2*L+1 & N_ph=2*L+1
    h_L_th, h_L_ph = 127/2, 11/2

    #Ensure sample number is integral
    h_N_th, h_N_ph = int_prec(2*(h_L_th)+1), int_prec(2*(h_L_ph)+1)

    #Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2*(h_N_th-1), 4*(h_N_th-1)
    h_N_ph_E = 2*h_N_ph

    #Extended domain lattice
    h_int_th = np.pi/(h_N_th-1)*np.arange(0, (h_N_th_EE-1)+1)
    h_int_ph = 2*np.pi/h_N_ph*np.arange(0, h_N_ph_E)
    
    #Evaluate function on 2-sphere
    h_int_fun_a_N_S2 = h_int_fun_a(h_int_th[:h_N_th], h_int_ph[:h_N_ph])
    h_int_fun_b_N_S2 = h_int_fun_b(h_int_th[:h_N_th], h_int_ph[:h_N_ph])
    
    print('=>half-integer: fwd')
    #Compute coefficients
    h_sf_arr = np.array([h_int_fun_a_N_S2, h_int_fun_b_N_S2])
    h_salm_arr = h_int_salm(h_s_arr, h_sf_arr)

    #Throw small values
    #h_salm_arr[np.abs(h_salm_arr)<1e-12] = 0
    
    #Indexing map:
    h_ind_map = lambda l2,m2: int_prec(1/2*(m2+l2*(l2/2+1)-1/2))
    #print(h_salm_arr[1][h_ind_map(11,11)])


    print('=>half-integer: bwd')
    h_sf_rec_arr = h_int_sf(h_s_arr, h_salm_arr, h_N_th, h_N_ph)
    
    #Do comparison
    h_num_fun = h_salm_arr.shape[0]
    print('>Error inspection')
    for f in np.arange(0, h_num_fun):
        print('sf->salm->sf_til Err @ fun=' + str(f) +' ' + \
            str(np.max(np.abs(h_sf_arr[f]- h_sf_rec_arr[f]))))


    print('=>half-integer: (rand fwd/bwd pairs)')
    h_s_r_arr = np.array([-5/2, -3/2, -1/2, 1/2, 3/2, 5/2])
    h_salm_r_arr = h_int_gen_rand_salm(h_s_r_arr, h_L_th, h_L_ph, h_L_th)
    h_sf_r_rec_arr = h_int_sf(h_s_r_arr, h_salm_r_arr, h_N_th, h_N_ph)
    h_salm_r_rec_arr = h_int_salm(h_s_r_arr, h_sf_r_rec_arr)

    h_num_fun_r = h_salm_r_arr.shape[0]

    print('>Error inspection')
    for f in np.arange(0, h_num_fun_r):
        print('s:' + str(h_s_r_arr[f]) + \
            ' h_salm->h_sf->h_salm_til Err @ ' + \
            str(np.max(np.abs(h_salm_r_arr[f] - h_salm_r_rec_arr[f]))))




    ###################################
    #Half-integer->Integer [CG product]
    ###################################

    #Specify a number of points that will form a common lattice on S2
    #for integer and half integer quantities

    N_th, N_ph = 50, 50
    h_N_th, h_N_ph = N_th, h_N_ph

    #Infer accessible band-limits    
    L_th, L_ph = int_prec(N_th/2-2), int_prec((N_ph-2)/2)
    h_L_th, h_L_ph = (h_N_th-1)/2, (h_N_ph-1)/2

    print([L_th, L_ph, h_L_th, h_L_ph])

    #Spin-weights
    h_s_arr = np.array([1/2, -3/2])
    #Test functions [build from analytical defn. of Wigner D]
    h_int_fun_a = lambda th, ph: swsh_an.sylm(h_s_arr[0],1/2,-1/2,th,ph) + \
        11/10*swsh_an.sylm(h_s_arr[0],3/2,1/2,th,ph)+ \
        0.1*swsh_an.sylm(h_s_arr[0],11/2,7/2,th,ph)

    h_int_fun_b = lambda th, ph: swsh_an.sylm(h_s_arr[1],3/2,-1/2,th,ph) + \
        -5/4*swsh_an.sylm(h_s_arr[1],11/2,11/2,th,ph)
    

    #Extension in th,ph requires more points
    h_N_th_E, h_N_th_EE = 2*(h_N_th-1), 4*(h_N_th-1)
    h_N_ph_E = 2*h_N_ph

    #Extended domain lattice
    h_int_th = np.pi/(h_N_th-1)*np.arange(0, (h_N_th_EE-1)+1)
    h_int_ph = 2*np.pi/h_N_ph*np.arange(0, h_N_ph_E)
    
    #Evaluate function on 2-sphere
    h_int_fun_a_N_S2 = h_int_fun_a(h_int_th[:h_N_th], h_int_ph[:h_N_ph])
    h_int_fun_b_N_S2 = h_int_fun_b(h_int_th[:h_N_th], h_int_ph[:h_N_ph])
    
    #Pointwise products
    int_fun_a_N_S2 = h_int_fun_a_N_S2*h_int_fun_a_N_S2
    int_fun_b_N_S2 = h_int_fun_a_N_S2*h_int_fun_b_N_S2

    #New spin-weights
    s_arr = int_prec( \
        np.array([h_s_arr[0]+h_s_arr[0], h_s_arr[0]+h_s_arr[1]]))

    #Compute coefficients
    sf_arr = np.array([int_fun_a_N_S2, int_fun_b_N_S2])
    salm_arr = int_salm(s_arr, sf_arr)




def JoergEmail(L_th, L_ph):
    '''
    
    '''

    print('=> test for Joerg')
    ###############################################
    # Test for Joerg:
    #
    # Specify the following CF:
    # CF(r,th) = cos(r)*(4+3*cos(2*th))
    #
    # Define the fcn:
    # f0 = \eth(CF)/[sqrt(2)*r CF**2]
    #
    # Now restrict s.t. r=1 and consider the cases:
    #
    ###############################################
    
    import matplotlib.pyplot as plt
    
    #Instantiate Fourier object
    FSi = FS.FourierSeries()
    SWSHsImn.FSi = FSi
    SWbwd.FSi = FSi
    
    ########
    #Integer
    ########

    #Indexing map:
    ind_map = lambda li,mi: li*(li+1)+mi
    
    #Num. nodes on S^2 grid [must be even]
    #To sample up to l=L /\ m=L take: N_th=2(L+2) & N_ph=2L+2
    #L_th, L_ph = 128, 2
    N_th, N_ph = 2*(L_th+2), 2*L_ph+2

    #Extension in theta direction requires more points
    N_th_E = 2*(N_th-1)

    #Extended domain lattice
    int_th = np.pi/(N_th-1)*np.arange(0, (N_th_E))
    int_ph = 2*np.pi/N_ph*np.arange(0, (N_ph-1)+1)
    
    CF = lambda r,th: np.cos(r)*(4+3*np.cos(2*th))
    eth_CF = lambda r,th: np.cos(r)*(-6*np.sin(2*th))

    CF_a = lambda th,ph: CF(1, th)
    CF_b = lambda th,ph: (1/CF(1, th))**2
    CF_c = lambda th,ph: eth_CF(1, th)
    

    s_arr = np.array([0, 0, 1, 1])
    
    #th,ph coords:
    th_S2, ph_S2 = np.meshgrid(int_th[:N_th], int_ph[:N_ph],indexing='ij')
    #Evaluate fcn on 2-sphere
    CF_a_N_S2 = np.array(CF_a(th_S2, ph_S2), dtype=np.complex128)
    CF_b_N_S2 = np.array(CF_b(th_S2, ph_S2), dtype=np.complex128)
    CF_c_N_S2 = np.array(CF_c(th_S2, ph_S2), dtype=np.complex128)

    f0_N_S2 = CF_c_N_S2/(np.sqrt(2))*CF_b_N_S2

    sf_arr = np.array([CF_a_N_S2, CF_b_N_S2, CF_c_N_S2, f0_N_S2])
    salm_arr = int_salm(s_arr, sf_arr)

    num_fun_r = salm_arr.shape[0]

    #Extract l=0 entries for plotting
    salm_arr_angular = np.zeros((num_fun_r, L_th+1),dtype=np.complex128)

    for f in np.arange(num_fun_r):
        for l in np.arange(L_th+1):
            salm_arr_angular[f,l] = salm_arr[f,ind_map(l,0)]

    
    #Eth ops.
    #raising: -sqrt((l-s)(l+s+1))
    #lowering: sqrt((l+s)(l-s+1))
    


    plt.figure(1)
    
    plt.semilogy(np.abs(salm_arr_angular[0]),'.r')
    plt.semilogy(np.abs(salm_arr_angular[1]),'.b')
    plt.semilogy(np.abs(salm_arr_angular[2]),'.g')
    plt.semilogy(np.abs(salm_arr_angular[3]),'.k')
    
    ax = plt.gca()
    ax.set_xlim(0, 128)
    plt.tight_layout()
    plt.xlabel(r'$l$',fontsize=12)
    plt.ylabel(r'$|{}_s a_{l,0}|$',fontsize=12)
    plt.tight_layout()
    
    
    return salm_arr_angular
    
if __name__ == '__main__':

    xform_prototype()

    #salm_arr_angular = JoergEmail(128, 2)


       
#
# :D
#
