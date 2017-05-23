#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ,-*
(_) Created on <Mon Jan 30 2017> @ 10:05:01

@author: Boris Daszuta
@function: Various unit tests of core functionality.
"""
import pytest as _pyt
import importlib as _il      # :( should release utilities too...

is_verbose = True            # passed to each test method
debug = False                # control drop to debug
imp_to_test = (              # control which .py are to be tested
    'test_error_pairs_int',
)

def imp_from_str(imp_as_str=None):
    '''
    Import based on input string.
    '''
    if imp_as_str is None:
        raise ValueError("'imp_as_str' must be a str.")

    try:
        _imp = _il.import_module(imp_as_str)
    except AttributeError:
        _imp = _sys.modules[imp_as_str]
    return _imp

def get_test_methods(imp):
    '''
    Given an import extract all methods with 'test_' prepend.
    '''
    lst_methods = dir(imp)
    use_meth = []
    for m in lst_methods:
        if 'test_' in m:
            if m.find('test_') == 0:
               use_meth.append(m)
    return tuple(use_meth)

def test_all(is_verbose=is_verbose):

    # prepare the method which are to be tested
    for imp in imp_to_test:
        imp_m = imp_from_str(imp_as_str=imp)       # import
        methods = get_test_methods(imp_m)          # methods to test

        for m_str in methods:                      # call each method
            m = getattr(imp_m, m_str)

            if is_verbose:
                print('=' * (len(m_str) + 1))
                print('@{}'.format(m_str))
                print('=' * (len(m_str) + 1))

            m(is_verbose=is_verbose)

def main(debug=True, is_verbose=True):
    _call_str = '-q '
    if debug:
        _call_str += '--pdb '
    else:
        pass
        # parallel exec with pytest-xdist
        # _call_str +=  '-n 4 '
    if is_verbose:
        _call_str += '-s --capture=no --duration=3 '
    _call_str += ' -v run_tests.py'

    _pyt.main(_call_str)

if __name__ == '__main__':
    main(debug=debug, is_verbose=is_verbose)

#
# :D
#
