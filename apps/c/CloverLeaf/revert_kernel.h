#ifndef REVERT_KERNEL_H
#define REVERT_KERNEL_H


void revert_kernel( const ACC<double> &density0, ACC<double> &density1,
                const ACC<double> &energy0, ACC<double> &energy1) {

  density1(0,0) = density0(0,0);
  energy1(0,0) = energy0(0,0);
}

#endif
