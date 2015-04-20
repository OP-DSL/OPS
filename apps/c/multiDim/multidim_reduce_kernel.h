#ifndef MULTIDIM_REDUCE_KERNEL_H
#define MULTIDIM_REDUCE_KERNEL_H

void multidim_reduce_kernel(const double *val, double *redu_dat1) {

  redu_dat1[0] = redu_dat1[0] + val[OPS_ACC_MD0(1,0,0)];
  redu_dat1[1] = redu_dat1[1] + val[OPS_ACC_MD0(2,0,0)];
}

#endif //MULTIDIM_REDUCE_KERNEL_H
