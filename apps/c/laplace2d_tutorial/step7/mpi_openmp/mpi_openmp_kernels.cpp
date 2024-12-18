// Auto-generated at 2024-12-18 22:21:36.499836 by ops-translator

// headers
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#include <limits>
#endif

//  global constants
extern int imax;
extern int jmax;
extern double pi;

void ops_init_backend(){}

// user kernel files
#include "set_zero_kernel.cpp"
#include "left_bndcon_kernel.cpp"
#include "right_bndcon_kernel.cpp"
#include "apply_stencil_kernel.cpp"
#include "copy_kernel.cpp"

