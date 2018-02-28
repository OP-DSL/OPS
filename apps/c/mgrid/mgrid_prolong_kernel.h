
#ifndef MGRID_PROLONG_KERNELS_H
#define MGRID_PROLONG_KERNELS_H

void mgrid_prolong_kernel(const double *coarse, double *fine, int *idx) {
  fine[OPS_ACC1(0,0)] = coarse[OPS_ACC0(0,0)];//10000*coarse[OPS_ACC0(-1,0)]+100*coarse[OPS_ACC0(0,0)]+coarse[OPS_ACC0(1,0)];
}



#endif //MGRID_PROLONG_KERNELS_H
