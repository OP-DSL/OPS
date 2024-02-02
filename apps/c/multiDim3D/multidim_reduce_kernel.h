#ifndef MULTIDIM_REDUCE_KERNEL_H
#define MULTIDIM_REDUCE_KERNEL_H

void multidim_reduce_kernel(const ACC<double> &val, double *redu_dat1) {

  redu_dat1[0] = redu_dat1[0] + val(0,0,0,0);
  redu_dat1[1] = redu_dat1[1] + val(1,0,0,0);
  redu_dat1[2] = redu_dat1[2] + val(2,0,0,0);
}

#endif //MULTIDIM_REDUCE_KERNEL_H
