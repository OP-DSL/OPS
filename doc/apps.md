
# OPS Example Applications

This directory contains a suite of scientific mini-applications and benchmarks ported to the OPS framework. Each application demonstrates how OPS enables single-source code to be automatically parallelized for a range of backends (OpenMP, CUDA, HIP, SYCL, MPI, etc.).

## How to Build and Run

- Each application has a `test.sh` script in its directory (e.g., `OPS/apps/c/CloverLeaf/test.sh`) that automates building and running all supported parallel versions.
- Set up the required environment variables (e.g., `OPS_INSTALL_PATH`, `MPI_INSTALL_PATH`, etc.) as described in the main OPS documentation.
- Run the test script: `bash test.sh`
- For more details on generated code, see the [OPS-APPS](https://github.com/OP-DSL/OPS-APPS) repository.

## Main Applications

### CloverLeaf (2D & 3D)
- **Location:** `apps/c/CloverLeaf`, `apps/c/CloverLeaf_3D`, `apps/c/CloverLeaf_3D_HDF5`
- **Purpose:** Solves the compressible Euler equations for hydrodynamics in 2D/3D using explicit, second-order methods. Used for benchmarking and performance studies. See the [original CloverLeaf project](https://uk-mac.github.io/) for more details.
- **Features:** Demonstrates all OPS backends, including MPI, OpenMP, CUDA, OpenCL, OpenACC, SYCL, and their combinations. The HDF5 variant tests file I/O.

### TeaLeaf
- **Location:** `apps/c/TeaLeaf`
- **Purpose:** Solves the linear heat conduction equation using iterative solvers (CG, Jacobi, Chebyshev, PPCG). Used for performance and scaling studies. See the [original TeaLeaf project](https://uk-mac.github.io/) for more details.
- **Features:** Supports all major OPS backends and demonstrates iterative solver patterns.

### ADI (Alternating Direction Implicit)

- **Location:** `apps/c/adi`
- **Purpose:** Demonstrates the ADI method for solving parabolic PDEs. Useful for testing OPS with implicit time-stepping and tridiagonal solvers. See the [original ADI/tridsolver project](https://github.com/OP-DSL/tridsolver) for more details.

## Additional C Applications

- **adi_burger / adi_burger_3D**: ADI method for Burger's equation in 2D/3D, demonstrating implicit solvers and tridiagonal systems.
- **compact_scheme**: Example of compact finite difference schemes.
- **complex / complex_numbers**: Demonstrates OPS support for complex data types and operations.
- **halfprecision**: Tests half-precision floating point support in OPS.
- **hdf5_slice**: Example of HDF5 file I/O and slicing.
- **laplace2d_tutorial**: Another Laplace equation tutorial in C.
- **mb_shsgc / mb_trid_test / mblock / mblock4D**: Multi-block and multi-dimensional domain decomposition and tridiagonal solver tests.
- **mgrid**: Multi-grid solver example.
- **multiDim / multiDim3D / multiDim_HDF5**: Multi-dimensional stencils and reductions, including HDF5 I/O.
- **ops-lbm**: Lattice Boltzmann Method (LBM) fluid dynamics solver.
- **random**: Random number generation and usage in OPS.
- **shsgc**: Scientific kernel or benchmark (details TBD).
- **tiling_fix**: Demonstrates tiling and performance tuning in OPS.
- **tti**: Tilted Transverse Isotropy (TTI) wave equation solver.
- **wave_test**: Wave equation solver.

## Fortran Applications

- **laplace2dtutorial**: Simple 2D Laplace equation solver (tutorial).
- **lowdim_test**: Low-dimensional test problems for regression and performance.
- **mblock**: Multi-block domain decomposition example.
- **multiDim / multiDim3D**: Multi-dimensional reduction and stencil operations.
- **poisson**: Poisson equation solver (also in C).
- **random**: Random number generation in Fortran with OPS.
- **shsgc**: Scientific kernel or benchmark (details TBD).
- **mathtest**: Mathematical function and operation tests (details TBD).
- **Purpose:** Solves the Poisson equation using iterative methods. Available in both C and Fortran, showing OPS support for multiple languages.

### Laplace2D Tutorial
- **Location:** `apps/fortran/laplace2dtutorial`
- **Purpose:** Simple 2D Laplace equation solver. Serves as a tutorial for new OPS users, especially in Fortran.

## Adding New Applications

- Place your application in the appropriate language subdirectory (`c/` or `fortran/`).
- Provide a `Makefile` to support building your application with the OPS build system.
- Provide a `test.sh` script to automate building and running all supported backends.
- Document any special requirements in a `README` file in your application directory.

---
For more information, see the main OPS documentation and the [OPS-APPS](https://github.com/OP-DSL/OPS-APPS) repository for pre-generated code and further examples.
