#ifndef REVERT_KERNEL_H
#define REVERT_KERNEL_H


void revert_kernel( const double *density0, double *density1,
                    const double *energy0,  double *energy1) {

  density1[OPS_ACC1(0,0,0)] = density0[OPS_ACC0(0,0,0)];
  energy1[OPS_ACC3(0,0,0)] = energy0[OPS_ACC2(0,0,0)];
}
#endif
