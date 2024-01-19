#!/bin/bash

voms-proxy-init --voms cms

source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh
ENVNAME="jmecoffteaenv"
source ../${ENVNAME}/bin/activate
export PYTHONPATH=${PWD}/../${ENVNAME}/lib/python3.6/site-packages:$PYTHONPATH
