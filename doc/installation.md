# Installation

**Note: The current CMakefile and relevant instructions are mainly tested on linux-based systems including Windows Subsystem for Linux**

## Dependencies

  * CMake

  CMake 3.18 or newer is required for using the CMake building system. If the latest version is not installed/shipped by default, it can be downloaded from https://cmake.org/download/, e.g., using the following script.
  ```bash
  version=3.19.0
  wget https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-Linux-x86_64.sh
  # Assume that CMake is going to be installed at /usr/local/cmake
  cmake_dir=/usr/local/cmake
  # sudo is not necessary for directories in user space.
  sudo mkdir $cmake_dir
  sudo sh ./cmake-$version-Linux-x86_64.sh --prefix=$cmake_dir  --skip-license
  sudo ln -s $cmake_dir/bin/cmake /usr/local/bin/cmake
  ```

  * Python2

  **Python2** is required by the OPS Python translator. The CMake build system will try to identify it automatically. However, the process can fail sometime (e.g., if there are both Python2 and Python3 installed). If this happens, the path to Python2 can be specified manually by using **-DPython2_EXECUTABLE** when invoking CMake

  * HDF5

  [HDF5](https://www.hdfgroup.org/solutions/hdf5) is required for parts of IO functionalities. The CMake build system **uses the parallel version by default** even for sequential codes, and automatically identify the library. If the automatic process fails, the path to the parallel HDF5 library can be specified by using -DHDF5_ROOT.

  * CUDA

  The CMake build system will detect the tookit automatically. If the automatic process fails, the build system will compile the library without the CUDA support.  please use -DCUDA_TOOLKIT_ROOT_DIR to manually specify the path.

<!-- 1. Set up environmental variables:

  * `CUDA_PATH` - Installation directory of CUDA, usually `/usr/local/cuda` (to build CUDA libs and applications, only needed if CUDA cannot be found in standard locations, or to enable OpenCL)
  * `MPI_HOME` - Installation directory of MPI (to build MPI based distributed memory libs and applications) only needed if MPI not installed in standard locations
  * `HDF5_ROOT` - Installation directory of HDF5 (to support HDF5 based File I/O) if HDF5 not installed in standard location -->

## Obtaining OPS

## Build OPS back-end libraries example applications
### Using `cmake`
#### Build the library and example applications together

  Create a build directory, and run CMake (version 3.18 or newer)
  ```bash
  mkdir build
  cd build
  # Please see below for CMake options
  cmake ${PATH_TO_OPS} -DBUILD_OPS_APPS=ON -DOPS_TEST=ON -DAPP_INSTALL_DIR=$HOME/OPS-APP -DCMAKE_INSTALL_PREFIX=$HOME/OPS-INSTALL -DGPU_NUMBER=1
  make # IEEE=1 this option is important for applications to get accurate results
  make install # sudo is needed if a directory like /usr/local/ is chosen.
  ```
After installation, the library and the python translator can be found at the direcory specified by CMAKE_INSTALL_PREFIX, together with the executable files for applications at APP_INSTALL_DIR.

####  Build the library and example applications separately

In this mode, the library can be firstly built and installed as

```bash
  mkdir build
  cd build
  # Please see below for CMake options
  cmake ${PATH_TO_OPS}   -DCMAKE_INSTALL_PREFIX=$HOME/OPS-INSTALL
  make # IEEE=1 this option is important for applications to get accurate results
  make install # sudo is needed if a system direction is chosen,
  ```
then the application can be built as

```bash
  mkdir appbuild
  cd appbuild
  # Please see below for CMake options
  cmake ${PATH_TO_APPS} -DOPS_INSTALL_DIR=$HOME/OPS-INSTALL -DOPS_TEST=ON -DAPP_INSTALL_DIR=$HOME/OPS-APP -DGPU_NUMBER=1
  make # IEEE=1 this option is important for applications to get accurate results
  ```
#### Tests

A few tasks for testing codes can be run by
```bash
  make test
  ```
The current tests are mainly based on the applications.
#### `cmake` options

  * `-DCMAKE_BUILD_TYPE=Release` - enable optimizations
  * `-DBUILD_OPS_APPS=ON` - build example applications (Library CMake only)
  * `-DOPS_TEST=ON` - enable the tests
  * `-DCMAKE_INSTALL_PREFIX=` - specify the installation direction for the library (/usr/local by default, Library CMake only)
  * `-DAPP_INSTALL_DIR=` - specify the installation direction for the applications ($HOME/OPS-APPS by default)
  * `-DGPU_NUMBER=` - specify the number of GPUs used in the tests
  * `-DOPS_INSTALL_DIR=` - specify where the OPS library is installed (Application CMake only, see [here](#build-the-library-and-example-applications-separately))
  * `-DOPS_VERBOSE_WARNING=ON` - show verbose output during building process
  <!-- * `-DHDF5_PREFER_PARALLEL=ON` - build using parallel HDF5, rather than serial HDF5 libraries -->
  <!-- * `-DBUILD_OPS_FROTRAN=ON` - enable building OPS Fortran libraries. -->

### Using regular `Makefiles`
#### Build library
#### Build application
#### Makefile options

## Running example applications
### CloverLeaf
### CloverLeaf_3D_HDF5
### poisson
### adi

## Runtime flags and options
