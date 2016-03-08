#ifndef TEST_KERNEL_H
#define TEST_KERNEL_H

#include "vars.h"


//void test_kernel(const double* rho_new), const double* readvar, double *rms) {
void test_kernel(const double *rho_new, double *rms) {
  //rms[0] = rms[0] + pow ((rho_new[OPS_ACC0(0)] - readvar[OPS_ACC1(0)]), 2);
  rms[0] = rms[0] + pow (rho_new[OPS_ACC0(0)], 2.0);
}
#endif
