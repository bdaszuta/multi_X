#!/bin/sh

##################################
# clean up temporary files / cache
##################################

find . -type d -name '__pycache__' -exec rm -r {} +
find . -type d -name '.cache' -exec rm -r {} +
