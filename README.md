##OPS

OPS is an API with associated libraries and pre-processors to generate
parallel executables for applications on multi-block structured grids.


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

  * `CUDA_PATH` - Installation directory of CUDA, usually `/usr/local/cuda` (to build CUDA libs and applications, only needed if CUDA cannot be found in standard locations, or to enable OpenCL)
  * `MPI_HOME` - Installation directory of MPI (to build MPI based distributed memory libs and applications) only needed if MPI not installed in standard locations
  * `HDF5_ROOT` - Installation directory of HDF5 (to support HDF5 based File I/O) if HDF5 not installed in standard location


2. Build OPS back-end libraries.


  Create a build directory, and run CMake (version 3.9 or newer)
  ```
  mkdir build
  cd build
  cmake ${PATH_TO_OPS}
  make
  ```

  Options of interest to specify to `cmake` include:

  * `-DCMAKE_BUILD_TYPE=Release` - enable optimizations
  * `-DBUILD_OPS_FROTRAN=ON` - enable building OPS Fortran libraries.
  * `-DBUILD_OPS_APPS=ON` - build example applications
  * `-DHDF5_PREFER_PARALLEL=ON` - build using parallel HDF5, rather than serial HDF5 libraries

