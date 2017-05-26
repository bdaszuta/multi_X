from multi_X_SWSH._SWSH._spectral import interface_Fourier_series
from multi_X_SWSH._SWSH._spectral import SWSH_periodic_extension
from multi_X_SWSH._SWSH._spectral import (SWSH_Imn, SWSH_salm, SWSH_bwd,
                                          SWSH_eth)

# analytical representation for comparison
from multi_X_SWSH._SWSH._special.wigner_d_AnSum import wigner_d_an

# TN style algorithm
from multi_X_SWSH._SWSH._special import (wigner_d_TN, SWSH_AnSum)

from multi_X_SWSH._types import *

# instantiate requisite internal Fourier object
_FSi = interface_Fourier_series.FourierSeries()
SWSH_Imn.FSi = SWSH_bwd.FSi = _FSi
