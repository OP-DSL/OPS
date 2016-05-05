#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

#include "data.h"
void init_kernel(double *val, int *idx){
  if(idx[0]==0 || idx[0]==nx-1 || idx[1]==0 || idx[1]==ny-1 || idx[2]==0 || idx[2]==nz-1)
    val[OPS_ACC0(0,0,0)] = 1.0;
  else
    val[OPS_ACC0(0,0,0)] = 0.0;
}
#endif //INIT_KERNEL_H
