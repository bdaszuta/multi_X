#!/bin/sh

######################################################
# Automatically run unit tests associated with project
#
# Requires:
#  See requirements_doc.txt
#
# Usage:
#  Run from base of repository
######################################################
PYTHONPATH=$PWD                                    # required to locate files
export PYTHONPATH

cd doc
make html
cd ..
