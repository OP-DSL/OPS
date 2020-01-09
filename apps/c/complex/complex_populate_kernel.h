#ifndef COMPLEX_POPULATE_KERNELS_H
#define COMPLEX_POPULATE_KERNELS_H

void complex_populate_kernel(complexd *val, complexd *red, int *idx) {
  val[OPS_ACC0(0,0)].real((double)idx[0]);
  val[OPS_ACC0(0,0)].imag((double)idx[1]);
  *red += val[OPS_ACC0(0,0)];
}

#endif //COMPLEX_POPULATE_KERNELS_H
