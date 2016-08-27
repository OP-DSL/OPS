#ifndef TEA_LEAF_KERNEL_H
#define TEA_LEAF_KERNEL_H

#include "data.h"
#include "definitions.h"

void tea_leaf_init_zero2_kernel (double * p, double * z) {
  p[OPS_ACC0(0,0)] = 0.0;
  z[OPS_ACC1(0,0)] = 0.0;
}

void tea_leaf_init_zero_kernel (double * p) {
  p[OPS_ACC0(0,0)] = 0.0;
}

void tea_leaf_yeqx_kernel (double * p, const double * x) {
  p[OPS_ACC0(0,0)] = x[OPS_ACC1(0,0)];
}
void tea_leaf_yeqax_kernel (double * p, const double * x, const double * a) {
  p[OPS_ACC0(0,0)] = x[OPS_ACC1(0,0)] * (*a);
}
void tea_leaf_dot_kernel (const double * r, const double * p, double *rro) {
  *rro = *rro + r[OPS_ACC0(0,0)] * p[OPS_ACC1(0,0)];
}

void tea_leaf_axpy_kernel(double * u, const double * p, const double * alpha) {
  u[OPS_ACC0(0,0)] = u[OPS_ACC0(0,0)] + (*alpha)*p[OPS_ACC1(0,0)];
}

void tea_leaf_axpby_kernel(double * u, const double * p, const double * alpha, const double * beta) {
  u[OPS_ACC0(0,0)] = (*alpha) * u[OPS_ACC0(0,0)] + (*beta)*p[OPS_ACC1(0,0)];
}

void tea_leaf_zeqxty_kernel(double * z, const double * x, const double * y) {
  z[OPS_ACC0(0,0)] = x[OPS_ACC1(0,0)] * y[OPS_ACC2(0,0)];
}

void tea_leaf_recip_kernel(double * u, const double * p) {
  u[OPS_ACC0(0,0)] = 1.0/p[OPS_ACC1(0,0)];
}

void tea_leaf_recip2_kernel(double *z, const double *x, const double *y) {
	z[OPS_ACC0(0,0)] = x[OPS_ACC1(0,0)]/y[OPS_ACC1(0,0)];
}

void tea_leaf_norm2_kernel(const double *x, double * norm) {
	*norm = *norm + x[OPS_ACC0(0,0)]*x[OPS_ACC0(0,0)];
}
#endif
