#!/bin/bash

exe=$1
target=$2
gpu_number=$3

if [[ "$target" == "seq" ]]; then
    echo "$exe"
elif [[ "$target" == "openmp" ]]; then
    echo "$exe"
elif [[ "$target" == "tiled" ]]; then
    echo "$exe OPS_TILING"
fi
