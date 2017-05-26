"""
 ,-*
(_) Created on <Fri May 26 2017> @ 12:22:21

@author: Boris Daszuta
@function: Collect common documentation replacements.
"""
REPLACEMENTS = {
    '$int': 'int',
    '$arr': 'array-like',
    '$bool': 'bool',
    '$L_th': 'The band-limit to use in the $m_th direction.',
    '$L_ph': 'The band-limit to use in the $m_ph direction.',
    '$is_half_integer': ('Specify what type of parameters '
                         'are in use.'),
    '$is_extended': 'Control whether extended data is worked with.',
    '$is_double_extension': ('Control whether doubly-extended data is '
                             'worked with.'),
    '$m_l': r':math:`l`',
    '$m_m': r':math:`m`',
    '$m_th': r':math:`\vartheta`',
    '$m_ph': r':math:`\varphi`',
    '$m_S2': r':math:`\mathbb{S}^2`',
    '$m_N_th': r':math:`N_\vartheta`',
    '$m_N_ph': r':math:`N_\varphi`'}


def _rep_doc(obj):
    doc = obj.__doc__
    changed = True
    while changed:
        changed = False
        for find_val, rep_val in REPLACEMENTS.items():
            if find_val in doc:
                doc = doc.replace(find_val, rep_val)
                changed = True

    obj.__doc__ = doc

#
# :D
#
