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
elif [[ "$target" == "cuda" ]]; then
    echo "$exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_cuda" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_cuda_tiled" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 OPS_TILING"
fi
