#!/bin/bash

exe=$1
target=$2
gpu_number=$3

if [[ "$target" == "dev_seq" ]]; then
    echo "$exe"
elif [[ "$target" == "dev_mpi" ]]; then
    echo "mpirun -np 4 --oversubscribe $exe"
elif [[ "$target" == "seq" ]]; then
    echo "$exe"
elif [[ "$target" == "openmp" ]]; then
    echo "$exe"
elif [[ "$target" == "tiled" ]]; then
    echo "$exe OPS_TILING"
elif [[ "$target" == "mpi" ]]; then
    echo "mpirun -np 4 --oversubscribe $exe"
elif [[ "$target" == "mpi_openmp" ]]; then
    echo "mpirun -np 4 --oversubscribe $exe"
elif [[ "$target" == "mpi_tiled" ]]; then
    echo "mpirun -np 4 --oversubscribe $exe OPS_TILING"
elif [[ "$target" == "cuda" ]]; then
    echo "$exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_cuda" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_cuda_tiled" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 OPS_TILING"
elif [[ "$target" == "ompoffload" ]]; then
    echo "$exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_ompoffload" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4"
elif [[ "$target" == "mpi_ompoffload_tiled" ]]; then
    echo "mpirun -np $gpu_number --oversubscribe $exe OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4 OPS_TILING"
fi

if [[ "$target" == "h5_dev_seq" ]]; then
    echo "mv $exe/tmp/output.h5 $exe/tmp/output_ref_seq.h5 && echo \"PASSED\" || echo \"FAILED\""
elif [[ "$target" == "h5_dev_mpi" ]]; then
    echo "mv $exe/tmp/output.h5 $exe/tmp/output_ref_mpi.h5 && echo \"PASSED\" || echo \"FAILED\""
elif [[ "$target" == "h5_seq" || "$target" == "h5_openmp" || "$target" == "h5_tiled" || "$target" == "h5_cuda" ]]; then
    echo "h5diff -p 1e-14 $exe/tmp/output.h5 $exe/tmp/output_ref_seq.h5 | { read output; [ -z \"$output\" ] && echo \"PASSED\" || echo \"FAILED\"; }"
elif [[ "$target" == "h5_mpi" || "$target" == "h5_mpi_openmp" || "$target" == "h5_mpi_tiled" ]]; then
    echo "h5diff -p 1e-14 $exe/tmp/output.h5 $exe/tmp/output_ref_mpi.h5 | { read output; [ -z \"$output\" ] && echo \"PASSED\" || echo \"FAILED\"; }"
fi
