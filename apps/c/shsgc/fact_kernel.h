#ifndef FACT_KERNEL_H
#define FACT_KERNEL_H

#include "vars.h"


void fact_kernel(const double* eff, double *s) {
  double fact;
  for (int m=0; m < 3 ;m++) {
    fact  = 0.50 * dt / dx ;
    s[OPS_ACC_MD1(m,0)] = -fact * (eff[OPS_ACC_MD0(m,0)] - eff[OPS_ACC_MD0(m,-1)]);
  }
}

#endif