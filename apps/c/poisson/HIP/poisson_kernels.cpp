//
// auto-generated by ops.py//

//header
#include <hip/hip_runtime.h>
#define OPS_API 2
#define OPS_2D
#include "ops_lib_core.h"

#include "ops_hip_rt_support.h"
#include "ops_hip_reduction.h"


#define OPS_FUN_PREFIX __device__ __host__
#include "user_types.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif
// global constants
#define dx dx_OPSCONSTANT
__constant__ double dx;
#define dy dy_OPSCONSTANT
__constant__ double dy;

void ops_init_backend() {}

//Dummy kernel to make sure constants are not optimized out
__global__ void ops_internal_this_is_stupid() {
((int*)&dx)[0]=0;
((int*)&dy)[0]=0;
}

void ops_decl_const_char(int dim, char const *type,
int size, char *dat, char const *name){
  ops_execute(OPS_instance::getOPSInstance());
  if (!strcmp(name,"dx")) {
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dx_OPSCONSTANT), dat, dim*size));
  }
  else
  if (!strcmp(name,"dy")) {
    hipSafeCall(OPS_instance::getOPSInstance()->ostream(),hipMemcpyToSymbol(HIP_SYMBOL(dy_OPSCONSTANT), dat, dim*size));
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
  }
}


//user kernel files
#include "poisson_kernel_populate_hip_kernel.cpp"
#include "poisson_kernel_update_hip_kernel.cpp"
#include "poisson_kernel_initialguess_hip_kernel.cpp"
#include "poisson_kernel_stencil_hip_kernel.cpp"
#include "poisson_kernel_error_hip_kernel.cpp"