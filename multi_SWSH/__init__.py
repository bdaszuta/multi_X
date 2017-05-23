from . import _SWSH
from ._SWSH._spectral.SWSH_eth import (eth_build, )
from ._interface import (sf_to_salm, salm_to_sf, generate_random_salm)
from ._types import *

# provide some idx / sz / grid convenience methods
from ._SWSH.idx_arr import *
from ._SWSH.grid import *
