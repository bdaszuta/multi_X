#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on Thu Dec 26 17:20:15 2013

@author: Boris Daszuta
@function: Direct translation of analytical expression for wigner_d.
"""
import numpy as _np


def _lnfactorial(num):
    '''
    Calculate a factorial using exp/ln trick - for integer values only
    '''
    def _lnfactorialcalc(num):
        # Ensure positive integer
        num = _np.round(num)
        num_ran = _np.arange(num, 0, -1)
        ln_num_ran = _np.log(num_ran)

        return _np.sum(ln_num_ran)

    vfunc_lnfactorialcalc = _np.vectorize(_lnfactorialcalc)
    return vfunc_lnfactorialcalc(num)


def wigner_d_an(l, m, n, theta):
    '''
    Reduced Wigner matrix 'd' - Analytical defn. which can be used for testing.

    Parameters
    ----------
    l : number
    m : number
    n : number
    theta : array-like

    Returns
    -------
    d^l_mn(theta)
    '''
    r_min = _np.round(_np.max(_np.array([0, m - n])))
    r_max = _np.round(_np.min(_np.array([l + m, l - n])))

    # Range of summation
    r = _np.arange(r_min, r_max + 1)
    # Phases
    ph = _np.power(-1, r - m + n)

    # Calculate factorials
    facts = 0.5 * (
        _lnfactorial(l + m) +
        _lnfactorial(l - m) +
        _lnfactorial(l + n) +
        _lnfactorial(l - n)) \
        - (
        _lnfactorial(r) +
        _lnfactorial(l + m - r) +
        _lnfactorial(l - r - n) +
        _lnfactorial(r - m + n))
    facts = _np.exp(facts)

    # Ensure we have correct i_nput types
    theta = _np.asarray(theta)
    theta_sz = theta.size
    r_sz = r.size

    costh_2 = _np.cos(theta / 2)
    sinth_2 = _np.sin(theta / 2)

    costh_2 = _np.tile(costh_2, (r_sz, 1))
    sinth_2 = _np.tile(sinth_2, (r_sz, 1))
    rt = _np.tile(r.reshape((r_sz, 1)), (1, theta_sz))

    va = ph * facts
    va = _np.tile(va.reshape((r_sz, 1)), (1, theta_sz))

    trigf = _np.power(costh_2, 2 * l - 2 * rt + m - n) * \
        _np.power(sinth_2, 2 * rt - m + n)

    theta = _np.tile(theta, (r_sz, 1))
    r = _np.tile(r.reshape((r_sz, 1)), (1, theta_sz))
    return _np.sum(va * trigf, 0)


#
# :D
#
