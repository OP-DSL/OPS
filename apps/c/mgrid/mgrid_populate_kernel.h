#ifndef MGRID_POPULATE_KERNELS_H
#define MGRID_POPULATE_KERNELS_H

void mgrid_populate_kernel_1(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+6*idx[1]);
}

void mgrid_populate_kernel_2(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+4*idx[1]);
}

void mgrid_populate_kernel_3(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+24*idx[1]);
}

#endif //MGRID_POPULATE_KERNELS_H
