#ifndef FACT_KERNEL_H
#define FACT_KERNEL_H

#include "vars.h"


void fact_kernel(const ACC<double>& eff, ACC<double>& s) {
  double fact;
  for (int m=0; m < 3 ;m++) {
    fact  = 0.50 * dt / dx ;
    s(m,0) = -fact * (eff(m,0) - eff(m,-1));
  }
}
#endif
