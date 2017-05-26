#!/bin/sh

##################################
# clean up temporary files / cache
##################################

find . -type d -name '__pycache__' -exec rm -rf {} +
find . -type d -name '.cache' -exec rm -rf {} +
find . -type d -name '_build' -exec rm -rf {} +
find . -type d -name '_static' -exec rm -rf {} +
find . -type d -name '_templates' -exec rm -rf {} +
