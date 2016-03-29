##OPS

OPS is an API with associated libraries and pre-processors to generate
parallel executables for applications on mulit-block structured grids.


This repository contains the implementation of the run-time library
and the pre-processor, and is structured as follows:

* ops: Implementation of the user and run-time OPS C/C++ APIs

* apps: Application examples in C.
  These are examples of user application code and also include
  the target code an OPS pre-processor should produce to correctly
  use the OPS run-time library.
  Currently the main application developed with OPS is a single
  block structured mesh application - Cloverleaf originally
  developed at https://github.com/Warwick-PCAV/CloverLeaf

* translator: Python OPS pre-processor for C/C++ API

* doc: Documentation

####Installation

1. Set up environmental variables:

  * `OPS_COMPILER` - compiler to be used (Currently supports Intel, PGI and Cray compilers, but others can be easily incorporated by extending the Makefiles used in step 2 and 3)
  * `OPS_INSTALL_PATH` - Installation directory of `OPS/ops`
  * `CUDA_INSTALL_PATH` - Installation directory of CUDA, usually `/usr/local/cuda` (to build CUDA libs and applications)
  * `OPENCL_INSTALL_PATH` - Installation directory of OpenCL, usually `/usr/local/cuda` for NVIDIA OpenCL implementation (to build OpenCL libs and applications)
  * `MPI_INSTALL_PATH` - Installation directory of MPI (to build MPI based distributed memory libs and applications)
  * `HDF5_INSTALL_PATH` - Installation directory of HDF5 (to support HDF5 based File I/O)

  See example scripts (e.g. source_intel, source_pgi_15.10, source_cray) under `OPS/ops/` that
  sets up the environment for building with various compilers (Intel, PGI, Cray).

2. Build OPS back-end libraries.

  For C/C++ back-end use Makefile under `OPS/ops/c` (modify Makefile if required). The libraries will be built in `OPS/ops/c/lib`
  ```
  cd $OPS_INSTALL_PATH/c
  make

  ```
  For Fortran back-end use Makefile under `OPS/ops/fortran` (modify Makefile if required). The libraries will be built in `OPS/ops/fortran/lib`
  ```
  cd $OPS_INSTALL_PATH/fortran
  make
  ```


3. Build OPS example applications

  For example to build CloverLeaf_3D under `OPS/apps/c/CloverLeaf_3D`
  ```
  cd ../apps/c/Cloverleaf_3D/
  make
  ```
