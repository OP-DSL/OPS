//
// auto-generated by ops.py//

//header
#define OPS_2D
#define OPS_API 2
#include "ops_lib_core.h"
#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "ops_sycl_rt_support.h"
#include "ops_sycl_reduction.h"
// global constants
cl::sycl::buffer<int,1> *imax_p=nullptr;
extern int imax;
cl::sycl::buffer<int,1> *jmax_p=nullptr;
extern int jmax;
cl::sycl::buffer<double,1> *pi_p=nullptr;
extern double pi;

void ops_init_backend() {}

void ops_decl_const_char(int dim, char const * type, int size, char * dat, char const * name ) {
  if (!strcmp(name,"imax")) {
    if (imax_p == nullptr) imax_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
    auto accessor = (*imax_p).get_access<cl::sycl::access::mode::write>();
    for ( int d=0; d<dim; d++ ){
      accessor[d] = ((int*)dat)[d];
    }
  }
  else
  if (!strcmp(name,"jmax")) {
    if (jmax_p == nullptr) jmax_p = new cl::sycl::buffer<int,1>(cl::sycl::range<1>(dim));
    auto accessor = (*jmax_p).get_access<cl::sycl::access::mode::write>();
    for ( int d=0; d<dim; d++ ){
      accessor[d] = ((int*)dat)[d];
    }
  }
  else
  if (!strcmp(name,"pi")) {
    if (pi_p == nullptr) pi_p = new cl::sycl::buffer<double,1>(cl::sycl::range<1>(dim));
    auto accessor = (*pi_p).get_access<cl::sycl::access::mode::write>();
    for ( int d=0; d<dim; d++ ){
      accessor[d] = ((double*)dat)[d];
    }
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
  }
}

//user kernel files
#include "set_zero_sycl_kernel.cpp"
#include "left_bndcon_sycl_kernel.cpp"
#include "right_bndcon_sycl_kernel.cpp"
#include "apply_stencil_sycl_kernel.cpp"
#include "copy_sycl_kernel.cpp"