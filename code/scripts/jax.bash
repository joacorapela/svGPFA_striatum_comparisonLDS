#!/bin/bash

SVGPFA_SRC=~/svGPFA/repos/svGPFA/
GCNU_COMMON_REPO=~/gatsby-swc/gatsby/code/gcnu_common_repo
CURRENT_DIR=`pwd`

source ~/.condaInit
conda activate svGPFAjax

cd $SVGPFA_SRC
git checkout jaxTrickSameNSpikes

cd $GCNU_COMMON_REPO
git checkout jax1

cd $CURRENT_DIR
