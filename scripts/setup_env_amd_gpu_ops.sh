#!/bin/bash

# CRAY
export OPS_COMPILER=amd
export OPS_INSTALL_PATH=/work/e01/e01/ashutoshl/8nov_senga_bugfix/OPS/ops

module purge

export AMD_ARCH=MI200

export PATH=/work/e01/e01/ashutoshl/install/amdflang/installation/rocm-afar-6405-drop-4.2.0/bin:$PATH
export LD_LIBRARY_PATH=/work/e01/e01/ashutoshl/install/amdflang/installation/rocm-afar-6405-drop-4.2.0/lib:$LD_LIBRARY_PATH

