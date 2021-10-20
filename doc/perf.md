# Performance Tuning

<!--## Vectorization-->

## Executing with GPUDirect

GPU direct support for MPI+CUDA, to enable (on the OPS side) add
**-gpudirect** when running the executable. You may also have to use
certain environmental flags when using different MPI distributions. For
an example of the required flags and environmental settings on the
Cambridge Wilkes2 GPU cluster see:\
<https://docs.hpc.cam.ac.uk/hpc/user-guide/performance-tips.html>
## Cache-blocking Tiling
OPS has a code generation (ops_gen_mpi_lazy) and build target for
tiling. Once compiled, to enable, use the `OPS_TILING` runtime parameter. This will look at the L3 cache size of your CPU and guess the correct
tile size. If you want to alter the amount of cache to be used for the
guess, use the ``OPS_CACHE_SIZE=XX`` runtime parameter, where the value is
in Megabytes. To manually specify the tile sizes, use the
``OPS_TILESIZE_X``, ``OPS_TILESIZE_Y``, and ``OPS_TILESIZE_Z`` runtime arguments.

When MPI is combined with OpenMP tiling can be extended to the MPI
halos. Set `OPS_TILING_MAXDEPTH` to increase the the halo depths so that
halos for multiple `ops_par_loops` can be exchanged with a single MPI
message (see [TPDS2017](https://ieeexplore.ieee.org/abstract/document/8121995) for more details)\
To test, compile CloverLeaf under ``OPS/apps/c/CloverLeaf``, modify clover.in
to use a $6144^2$ mesh, then run as follows:\
For OpenMP with tiling:
```bash
export OMP_NUM_THREADS=xx; numactl -physnodebind=0 ./cloverleaf_tiled OPS_TILING
```
For MPI+OpenMP with tiling:
```bash
export OMP_NUM_THREADS=xx; mpirun -np xx ./cloverleaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6
```
To manually specify the tile sizes (in number of grid points), use the
OPS_TILESIZE_X, OPS_TILESIZE_Y, and OPS_TILESIZE_Z runtime arguments:
```bash
export OMP_NUM_THREADS=xx; numactl -physnodebind=0 ./cloverleaf_tiled OPS_TILING OPS_TILESIZE_X=600 OPS_TILESIZE_Y=200
```
## OpenMP and OpenMP+MPI
It is recommended that you assign one MPI rank per NUMA region when executing MPI+OpenMP parallel code. Usually for a multi-CPU system a single CPU socket is a single NUMA region. Thus, for a 4 socket system, OPS's MPI+OpenMP code should be executed with 4 MPI processes with each MPI process having multiple OpenMP threads (typically specified by the `OMP_NUM_THREAD`s flag). Additionally on some systems using `numactl` to bind threads to cores could give performance improvements (see `OPS/scripts/numawrap` for an example script that wraps the `numactl` command to be used with common MPI distributions). 

## CUDA arguments
The CUDA (and OpenCL) thread block sizes can be controlled by setting
the ``OPS_BLOCK_SIZE_X``, ``OPS_BLOCK_SIZE_Y`` and ``OPS_BLOCK_SIZE_Z`` runtime
arguments. For example,
```bash
./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4
```

## OpenCL arguments
`OPS_CL_DEVICE=XX` runtime flag sets the OpenCL device to execute the
code on.

Usually `OPS_CL_DEVICE=0` selects the CPU and `OPS_CL_DEVICE=1` selects
GPUs.

