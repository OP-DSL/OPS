//
// auto-generated by ops.py
//

#include "./MPI_inline/complex_numbers_common.h"


void ops_init_backend() {}

void ops_decl_const_char2(int dim, char const *type,
int size, char *dat, char const *name){
  if (!strcmp(name,"c0")) {
    c0 = *(double*)dat;
  }
  else
  if (!strcmp(name,"rc0")) {
    rc0 = *(double*)dat;
  }
  else
  if (!strcmp(name,"rc1")) {
    rc1 = *(double*)dat;
  }
  else
  if (!strcmp(name,"rc2")) {
    rc2 = *(double*)dat;
  }
  else
  if (!strcmp(name,"rc3")) {
    rc3 = *(double*)dat;
  }
  else
  if (!strcmp(name,"nx0")) {
    nx0 = *(int*)dat;
  }
  else
  if (!strcmp(name,"deltai0")) {
    deltai0 = *(double*)dat;
  }
  else
  if (!strcmp(name,"deltat")) {
    deltat = *(double*)dat;
  }
  else
  {
    throw OPSException(OPS_RUNTIME_ERROR, "error: unknown const name");
  }
}

//user kernel files
#include "complex_numbers_block0_5_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_4_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_0_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_1_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_2_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_3_kernel_mpiinline_kernel.cpp"
#include "complex_numbers_block0_cn_kernel_mpiinline_kernel.cpp"