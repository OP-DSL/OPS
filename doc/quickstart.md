# OPS Quickstart Guide

Welcome to OPS! This guide will help you get a working OPS application running in minutes.

## 1. Clone the Repository

```bash
git clone https://github.com/OP-DSL/OPS.git
cd OPS
```

## 2. Set Up Your Environment

- Load required modules for your system (compiler, MPI, CUDA, etc.). Consult your cluster or system documentation for details.
- If a setup script is provided for your platform (see `source_files/`), source it. Otherwise, set environment variables (e.g., `OPS_INSTALL_PATH`, `CUDA_INSTALL_PATH`, `MPI_INSTALL_PATH`) as needed for your environment.
 The OPS code generator (`ops_translator`) requires a Python virtual environment. For the **Makefile build** it is created automatically under `ops_translator/.python/`; for the **CMake build** it is created under `${CMAKE_INSTALL_PREFIX}/translator/ops_translator/ops_venv/` during the configure step. No manual setup is needed in either case. For details and HPC-specific notes, see `doc/installation.md`.


## 3. Build Backend Libraries

Before building and running applications, you must build the OPS backend libraries for your target platforms. From the OPS root directory:

- For the C/C++ backend library:
	```bash
	cd ops/c
	make
	cd ../..
	```
- For the Fortran backend library (if needed):
	```bash
	cd ops/fortran
	make
	cd ../..
	```

Make sure your environment variables (e.g., OPS_INSTALL_PATH, CUDA_INSTALL_PATH, MPI_INSTALL_PATH) are set appropriately for your platform. See the example scripts in the `scripts/` directory for help with environment setup.

## 4. Build a Sample Application

Navigate to an example app, e.g. CloverLeaf:
```bash
cd apps/c/CloverLeaf
make
```
This will build all available versions (sequential, MPI, OpenMP, CUDA, etc.) for your environment. To run a specific version, execute the corresponding binary. For example:
```bash
./cloverleaf_seq                # Run the sequential version
mpirun -np 2 ./cloverleaf_mpi   # Run the MPI version
./cloverleaf_cuda               # Run the CUDA version
```
See the Makefile in each app directory for the list of available targets and executables.

## 5. Check Results

- Output and performance logs will be generated in the app directory.
- Check for "PASSED" in the output to confirm success.

## 6. Next Steps

- The above quickstart guide used the Makefile build system of OPS. For further detailed instructions including dependencies and the CMake-based build instructions please see [installation.md](installation.md).
- Explore other applications in [apps/c/](../apps/c/) and [apps/fortran/](../apps/fortran/).
- See [apps.md](apps.md) for details on each example.
- See [devdoc.md](devdoc.md) for developer and contributor information.

For more help, see the full documentation or open an issue on the OPS GitHub.