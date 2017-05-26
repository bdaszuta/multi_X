"""
 ,-*
(_) Created on <Tue May 23 2017> @ 09:30:59

@author: Boris Daszuta
@function: Transform specific settings.
"""
# keyword arguments that are fed to each JIT decorator.
_JIT_KWARGS = {'nopython': True, 'nogil': True, 'cache': True}

# maximum size to use with clru_cache
_FC_KWARGS = {'maxsize': None}

#
# :D
#
