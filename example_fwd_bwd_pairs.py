#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Tue May 23 2017> @ 08:01:20

@author: Boris Daszuta
@function: Example of SWSH transformation for (half) integer spin-weighted
fields ${}_sf$.

Here we perform the sequence of transformations: salm -> sf -> \tilde{salm}
and compare initial salm with the reconstructed \tilde{salm}.
"""
from numpy import (array, abs, max)
import multi_SWSH as ms


###############################################################################
# integer SWSH example
###############################################################################
I_s = array([-2, 1, 7])  # spin-weights
I_L_th, I_L_ph = 32, 47  # resolutions do not need to be equal

# infer what would be the number of samples using band-limits
I_N_th, I_N_ph = ms.L_to_N(L_th=I_L_th, L_ph=I_L_ph, is_half_integer=False)

# generate random coefficients
I_rand_salm = ms.generate_random_salm(s=I_s, L_th=I_L_th, L_ph=I_L_ph,
                                      is_half_integer=False)

# construct the associated spin-weighted functions
I_sf = ms.salm_to_sf(s=I_s, alm=I_rand_salm, N_th=I_N_th, N_ph=I_N_ph,
                     is_half_integer=False)

# reconstruct coefficients from the functions
I_salm = ms.sf_to_salm(s=I_s, f=I_sf, is_half_integer=False)

# retain maximum absolute error
I_err = max(abs(I_rand_salm - I_salm), axis=1)

###############################################################################
# half-integer SWSH example
# In the following only provide numerator ie if s = -3 / 2 and L = 7 / 2 then
# to a function pass -3 and 7.
###############################################################################
HI_s = array([-3, 1, 7])   # note: spin-weights
HI_s = array([-3,])
HI_L_th, HI_L_ph = 37, 11  # resolutions do not need to be equal

# infer what would be the number of samples using band-limits
HI_N_th, HI_N_ph = ms.L_to_N(L_th=HI_L_th, L_ph=HI_L_ph, is_half_integer=True)


# generate random coefficients
HI_rand_salm = ms.generate_random_salm(s=HI_s, L_th=HI_L_th, L_ph=HI_L_ph,
                                       is_half_integer=True)

# construct the associated spin-weighted functions
HI_sf = ms.salm_to_sf(s=HI_s, alm=HI_rand_salm, N_th=HI_N_th, N_ph=HI_N_ph,
                      is_half_integer=True)

# reconstruct coefficients from the functions
HI_salm = ms.sf_to_salm(s=HI_s, f=HI_sf, is_half_integer=True)

# retain maximum absolute error
HI_err = max(abs(HI_rand_salm - HI_salm), axis=1)


###############################################################################
# print some information
###############################################################################
print('@ integer:')
print('s={}'.format(I_s))
print('(L_th, L_ph)={}'.format((I_L_th, I_L_ph)))
print('max_abs_err={}'.format(I_err))

print('\n@ half-integer (s, L are doubled!):')
print('s={}'.format(HI_s))
print('(L_th, L_ph)={}'.format((HI_L_th, HI_L_ph)))
print('max_abs_err={}'.format(HI_err))


#
# :D
#
