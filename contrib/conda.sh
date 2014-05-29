#!/bin/sh
conda install -c https://conda.binstar.org/collections/yungyuc/solvcon \
  setuptools mercurial \
  scons cython numpy netcdf4 scotch nose gmsh vtk sphinx graphviz
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
easy_install -UZ sphinxcontrib-issuetracker
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi