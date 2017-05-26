"""
 ,-*
(_) Created on Thu Dec 26 17:19:17 2013

@author: Boris Daszuta
@function: One analytical defn. of the swsh
"""
import numpy as _np

from multi_X_SWSH._SWSH._special import wigner_d_AnSum


def sylm(s, l, m, theta, phi):
    '''
    >s ~ scalar
    >l ~ scalar
    >m ~ scalar
    >theta ~ numpy array
    >phi ~ numpy array

    <sYlm(theta,phi) ~ [num(theta) x num(phi)]
    '''
    wig_val = wigner_d_AnSum.wigner_d(l, s, m, theta)
    phi = _np.asarray(phi)
    azi_val = _np.exp(1j*m*phi)

    wig_val_sz = wig_val.size
    azi_val_sz = azi_val.size

    thv = _np.tile(_np.reshape(wig_val, (wig_val_sz, 1)), (1, azi_val_sz))
    phv = _np.tile(azi_val, (wig_val_sz, 1))

    return _np.sqrt((2*l + 1) / (4 * _np.pi)) * thv * phv


def _main():
    retval = sylm(1/2, 1/2, 1/2, [0, 2, 3], [0, 1])
    print(retval)

#
# :D
#
