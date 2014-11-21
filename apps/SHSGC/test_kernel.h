#ifndef VARS_KERNEL_H
#define VARS_KERNEL_H

#include "vars.h"


void test_kernel(const double* rho_new, const double* readvar, double *rms) {
  rms[0] = rms[0] + pow ((rho_new[OPS_ACC0(0)] - readvar[OPS_ACC1(0)]), 2);
}

#endif