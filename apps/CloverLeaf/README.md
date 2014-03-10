A hydrodynamics mini-app to solve the compressible Euler equations in 2D,
using an explicit, second-order method. See http://warwick-pcav.github.io/CloverLeaf/
for the original version and more details.

This version utilizes the OPS library and preprocessor. With OPS a single
application code is transformed to enable it to be run with a number of
parallelisations: OpenMP, CUDA and MPI and their combinations.


####Directory Structure

* MPI_Dev - Developer version that uses only a header file to get the application
running as a sequential application and with unoptimized MPI. Used for application
debugging before using the code generation tools of OPS.

* MPI - MPI and sequential version, code generated through the OPS translator. This
include platform specific optimisations such as verctorization

* MPI_OpenMP - MPI+OpenMP version, code generated through the OPS translator.

* MPI_CUDA - MPI+CUDA version, code generated through the OPS translator. (currently
there is no MPI support for this version)
