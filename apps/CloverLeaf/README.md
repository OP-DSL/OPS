A hydrodynamics mini-app to solve the compressible Euler equations in 2D,
using an explicit, second-order method. See http://warwick-pcav.github.io/CloverLeaf/
for the original version and more details.

This version utilizes the OPS library and preprocessor. With OPS a single
application code is transformed to enable it to be run with a number of
parallelisations: OpenMP, CUDA and MPI and their combinations.


###Directory Structure

* MPI_Dev - Developer version that uses only a header file to get the application
running with MPI (also the sequential version is folded into this directory)

* MPI - MPI and sequential version code generated through the OPS translator. This
include platform specific optimisations such as verctorization
