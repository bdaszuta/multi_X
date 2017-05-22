#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Thu Dec 26 17:20:15 2013

@author: Boris Daszuta
@function: Reduced Wigner matrix 'd' - Analytical defn.

See Num-I notebook pg. 28
"""

import numpy as np


def _lnfactorial(num):
    '''
    Calculate a factorial using exp/ln trick - for integer values only
    '''
    def _lnfactorialcalc(num):
        #Ensure positive integer
        num = np.round(num)
        num_ran = np.arange(num,0,-1)
        ln_num_ran = np.log(num_ran)

        return np.sum(ln_num_ran)

    vfunc_lnfactorialcalc = np.vectorize(_lnfactorialcalc)
    return vfunc_lnfactorialcalc(num)


def wigner_d(l,m,n,theta):
    '''
    >l ~ scalar number
    >m ~ scalar number
    >n ~ scalar number
    >theta ~ np array

    <d^l_mn(theta)
    '''
    r_min = np.round(np.max(np.array([0, m-n])))
    r_max = np.round(np.min(np.array([l+m, l-n])))

    #Range of summation
    r = np.arange(r_min, r_max+1)
    #Phases
    ph = np.power(-1, r-m+n)

    #Calculate factorials
    facts = 0.5*(
        _lnfactorial(l+m)+
        _lnfactorial(l-m)+
        _lnfactorial(l+n)+
        _lnfactorial(l-n)) \
        -(
        _lnfactorial(r)+
        _lnfactorial(l+m-r)+
        _lnfactorial(l-r-n)+
        _lnfactorial(r-m+n))
    facts = np.exp(facts)

    #Ensure we have correct input types
    theta = np.asarray(theta)
    theta_sz = theta.size
    r_sz = r.size

    costh_2 = np.cos(theta/2)
    sinth_2 = np.sin(theta/2)

    costh_2 = np.tile(costh_2,(r_sz,1))
    sinth_2 = np.tile(sinth_2,(r_sz,1))
    rt = np.tile(r.reshape((r_sz,1)), (1,theta_sz))

    va = ph*facts
    va = np.tile(va.reshape((r_sz,1)), (1,theta_sz))

    trigf = np.power(costh_2,2*l-2*rt+m-n)*np.power(sinth_2,2*rt-m+n)

    theta = np.tile(theta,(r_sz,1))
    r = np.tile(r.reshape((r_sz,1)), (1,theta_sz))

    #print(np.sum(va*trigf,0))

    return np.sum(va*trigf,0)


#dvals = wigner_d(5/2,1/2,-1/2,[0, 1])




#
# :D
#
