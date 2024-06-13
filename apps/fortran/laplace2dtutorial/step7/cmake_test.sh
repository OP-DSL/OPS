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
elif [[ "$target" == "mpi" ]]; then
    echo "mpirun -np 6 --oversubscribe $exe"
elif [[ "$target" == "mpi_openmp" ]]; then
    echo "mpirun -np 6 --oversubscribe $exe"
elif [[ "$target" == "mpi_tiled" ]]; then
    echo "mpirun -np 6 --oversubscribe $exe OPS_TILING"
fi
