# Getting Started

**Note: The current CMakefile and relevant instructions are mainly tested on linux-based systems including Windows Subsystem for Linux**

## Dependencies

The following prerequisites and dependencies are required for building OPS. Building each of the **backends** are optional and depends on the hardware and/or capabilities you will be targeting. 

  **CMake**

CMake 3.18 or newer is required for using the CMake building system. If the latest version is not installed/shipped by default, it can be downloaded from https://cmake.org/download/, e.g., using the following script.
  ```bash {r}
  version=3.19.0
  wget https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-Linux-x86_64.sh
  # Assume that CMake is going to be installed at /usr/local/cmake
  cmake_dir=/usr/local/cmake
  # sudo is not necessary for directories in user space.
  sudo mkdir $cmake_dir
  sudo sh ./cmake-$version-Linux-x86_64.sh --prefix=$cmake_dir  --skip-license
  sudo ln -s $cmake_dir/bin/cmake /usr/local/bin/cmake
  ```

 **Python**

The Python dependencies (primarily used for the OPS code generator) are best installed by setting up a virtual environment so that required packages can be installed without superuser privileges. To set up the Python virtual environment and install the required dependant packages, ensure that you have Python3.9 or a more recent version with pip installed.
Detailed instructions for installing virtual environment using pip can be found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
Execute following **after cloning the OPS repository (see below)** to install required packages. Note `OPS_INSTALL_PATH` is the installation directory of OPS/ops:
```
#Install virtual environment using pip (if not installed earlier)
python3 -m pip install --user virtualenv

mkdir -p $OPS_INSTALL_PATH/../ops_translator_v2/ops_venv
python3 -m venv $OPS_INSTALL_PATH/../ops_translator_v2/ops_venv
source $OPS_INSTALL_PATH/../ops_translator_v2/ops_venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install fparser cached-property dataclasses jinja2 pylint mypy pcpp sympy
python3 -m pip install clang==14.0.6 libclang==14.0.6
```
These instructions can be executed by running the script `OPS/ops_translator_v2/setup_venv.sh` file. 
After successfully setting up the Python virtual environment and installing the required dependent packages using the above instructions, you will need to activate the virtual environment by `source $OPS_INSTALL_PATH/../ops_translator_v2/ops_venv/bin/activate` every time you want to use the code generator. Activating the virtual environment ensures that the code generator and its dependencies are isolated from the system-wide Python installtion, avoiding conflicts and ensuring proper execution.

 **HDF5**

[HDF5](https://www.hdfgroup.org/solutions/hdf5) is required for parts of IO functionalities. The CMake build system **uses the parallel version by default** even for sequential codes, and automatically identify the library. If the automatic process fails, the path to the parallel HDF5 library can be specified by using `-DHDF5_ROOT`.

 **CUDA Backend**
 
The [CUDA](https://developer.nvidia.com/cuda-downloads) backend targets NVIDIA GPUs with a compute capability of 3.0 or greater. The CMake build system will detect the tookit automatically. If the automatic process fails, the build system will compile the library without the CUDA support.  Please use `-DCUDA_TOOLKIT_ROOT_DIR` to manually specify the path.

 **HIP Backend**

The HIP backend targets AMD GPUs and NVIDIA GPUs which are supported by HIP - either through its CUDA support or the [ROCm](https://rocmdocs.amd.com/en/latest/) stack (tested with >=3.9). 

 **SYCL Backend**

The [SYCL](https://www.khronos.org/sycl/) backend is currently in development and only working without MPI. It has been tested with Intel OneAPI (>=2021.1), Intel's public LLVM version, and hipSYCL (>=0.9.1), and runs on Intel CPUs and GPUs through Intel's OpenCL and Level Zero, NVIDIA and AMD GPUs both with the LLVM fork as well as hipSYCL. hipSYCL's OpenMP support covers most CPU architectures too.

 **Tridiagonal Solver Backend**

To use the tridiagonal solver OPS API in applications and build example applications such as `adi`, `adi_burger` and `adi_burger_3D` the open source tridiagonal solver (scalar) library needs to be cloned and built from the [Tridsolver repository](https://github.com/OP-DSL/tridsolver). 
```bash
git clone https://github.com/OP-DSL/tridsolver.git
```
Details on building scalar tridiagonal solver library can be found in the [README](https://github.com/OP-DSL/tridsolver/blob/master/scalar/README) file located at the appropriate subdirectory.

## Obtaining OPS
The latest OPS source code can be obtained by cloning the [OPS repository](https://github.com/OP-DSL/OPS) using
```bash
git clone https://github.com/OP-DSL/OPS.git
```
    
## Build OPS
### Using cmake
#### Build library and example applications together

  Create a build directory, and run CMake (version 3.18 or newer)
  ```bash
  mkdir build
  cd build
  # Please see below for CMake options
  cmake ${PATH_TO_OPS} -DBUILD_OPS_APPS=ON -DOPS_TEST=ON -DAPP_INSTALL_DIR=$HOME/OPS-APP -DCMAKE_INSTALL_PREFIX=$HOME/OPS-INSTALL -DGPU_NUMBER=1
  make # IEEE=1 enable IEEE flags in compiler
  make install # sudo is needed if a directory like /usr/local/ is chosen.
  ```
After installation, the library and the python translator can be found at the direcory specified by `CMAKE_INSTALL_PREFIX`, together with the executable files for applications at `APP_INSTALL_DIR`.

####  Build library and example applications separately

In this mode, the library can be firstly built and installed as

```bash
mkdir build
cd build
# Please see below for CMake options
cmake ${PATH_TO_OPS}   -DCMAKE_INSTALL_PREFIX=$HOME/OPS-INSTALL
make # IEEE=1 enable IEEE flags in compiler
make install # sudo is needed if a system direction is chosen,
```
Then the application can be built as:

```bash
mkdir appbuild
cd appbuild
# Please see below for CMake options
cmake ${PATH_TO_APPS} -DOPS_INSTALL_DIR=$HOME/OPS-INSTALL -DOPS_TEST=ON -DAPP_INSTALL_DIR=$HOME/OPS-APP -DGPU_NUMBER=1
make # IEEE=1 this option is important for applications to get accurate results
```
<!-- #### Tests

A few tasks for testing codes can be run by
```bash
make test
```
The current tests are mainly based on the applications.
-->

#### cmake options

  * `-DCMAKE_BUILD_TYPE=Release` - enable optimizations
  * `-DBUILD_OPS_APPS=ON` - build example applications (Library CMake only)
  * `-DOPS_TEST=ON` - enable the tests
  * `-DCMAKE_INSTALL_PREFIX=` - specify the installation direction for the library (`/usr/local` by default, Library CMake only)
  * `-DAPP_INSTALL_DIR=` - specify the installation direction for the applications (`$HOME/OPS-APPS` by default)
  * `-DGPU_NUMBER=` - specify the number of GPUs used in the tests
  * `-DOPS_INSTALL_DIR=` - specify where the OPS library is installed (Application CMake only, see [here](#build-the-library-and-example-applications-separately))
  * `-DOPS_VERBOSE_WARNING=ON` - show verbose output during building process
  <!-- * `-DHDF5_PREFER_PARALLEL=ON` - build using parallel HDF5, rather than serial HDF5 libraries -->
  <!-- * `-DBUILD_OPS_FROTRAN=ON` - enable building OPS Fortran libraries. -->

<!-- 1. Set up environmental variables:
* `CUDA_PATH` - Installation directory of CUDA, usually `/usr/local/cuda` (to build CUDA libs and applications, only needed if CUDA cannot be found in standard locations, or to enable OpenCL)
* `MPI_HOME` - Installation directory of MPI (to build MPI based distributed memory libs and applications) only needed if MPI not installed in standard locations
* `HDF5_ROOT` - Installation directory of HDF5 (to support HDF5 based File I/O) if HDF5 not installed in standard location -->

### Using Makefiles
#### Set up environmental variables:

  * `OPS_COMPILER` - compiler to be used (Currently supports Intel, PGI and Cray compilers, but others can be easily incorporated by extending the Makefiles used in step 2 and 3)
  * `OPS_INSTALL_PATH` - Installation directory of OPS/ops
  * `CUDA_INSTALL_PATH` - Installation directory of CUDA, usually `/usr/local/cuda` (to build CUDA libs and applications)
  * `OPENCL_INSTALL_PATH` - Installation directory of OpenCL, usually `/usr/local/cuda` for NVIDIA OpenCL implementation (to build OpenCL libs and applications)
  * `MPI_INSTALL_PATH` - Installation directory of MPI (to build MPI based distributed memory libs and applications)
  * `HDF5_INSTALL_PATH` - Installation directory of HDF5 (to support HDF5 based File I/O)

See example scripts (e.g. `source_intel`, `source_pgi_15.10`, `source_cray`) under `OPS/ops/scripts` that sets up the environment for building with various compilers (Intel, PGI, Cray).

#### Build back-end library
For C/C++ back-end use Makefile under `OPS/ops/c` (modify Makefile if required). The libraries will be built in `OPS/ops/c/lib`
```bash
cd $OPS_INSTALL_PATH/c
make
```
For Fortran back-end use Makefile under `OPS/ops/fortran` (modify Makefile if required). The libraries will be built in `OPS/ops/fortran/lib`
```bash
cd $OPS_INSTALL_PATH/fortran
make
```
#### Build exampe applications
For example to build CloverLeaf_3D under `OPS/apps/c/CloverLeaf_3D`
```bash  
cd ../apps/c/Cloverleaf_3D/
make
```  
<!---#### Makefile options -->


