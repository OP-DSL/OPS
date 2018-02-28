#ifndef MGRID_RESTRICT_KERNELS_H
#define MGRID_RESTRICT_KERNELS_H

void mgrid_restrict_kernel(const double *fine, double *coarse, int *idx) {
  //coarse[OPS_ACC1(0,0)] = 1000000*fine[OPS_ACC0(-1,0)]+1000*fine[OPS_ACC0(0,0)]+fine[OPS_ACC0(1,0)];
  coarse[OPS_ACC1(0,0)] = fine[OPS_ACC0(0,0)];
}
#endif //MGRID_PROLONG_KERNELS_H
