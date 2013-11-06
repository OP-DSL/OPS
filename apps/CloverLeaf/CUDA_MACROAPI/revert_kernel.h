#ifndef REVERT_KERNEL_H
#define REVERT_KERNEL_H

inline void revert_kernel( double *density0, double *density1,
                double *energy0, double *energy1) {

  density1[OPS_ACC1(0,0)] = density0[OPS_ACC0(0,0)];
  energy1[OPS_ACC3(0,0)] = energy0[OPS_ACC2(0,0)];
}
#endif
