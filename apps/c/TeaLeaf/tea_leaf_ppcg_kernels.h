#ifndef TEA_LEAF_PPCG_KERNEL_H
#define TEA_LEAF_PPCG_KERNEL_H

#include "data.h"
#include "definitions.h"


void tea_leaf_ppcg_init1_kernel(double *sd, double *rtemp, double *utemp, const double *z, const double *r, const double *theta_r) {
	sd[OPS_ACC0(0,0)] = z[OPS_ACC3(0,0)]*(*theta_r);
	rtemp[OPS_ACC1(0,0)] = r[OPS_ACC4(0,0)];
	utemp[OPS_ACC2(0,0)] = sd[OPS_ACC0(0,0)];
}

void tea_leaf_ppcg_init2_kernel(double *sd, double *rtemp, double *utemp, const double *r, const double *theta_r) {
	sd[OPS_ACC0(0,0)] = r[OPS_ACC3(0,0)]*(*theta_r);
	rtemp[OPS_ACC1(0,0)] = r[OPS_ACC3(0,0)];
	utemp[OPS_ACC2(0,0)] = sd[OPS_ACC0(0,0)];
}

void tea_leaf_ppcg_inner1_kernel(double *rtemp, const double *Kx, const double *Ky,
  const double *sd,const double *rx,const double *ry) {
	double smvp = 0.0;
  smvp = (1.0
    + (*ry)*(Ky[OPS_ACC2(0, 1)] + Ky[OPS_ACC2(0,0)])
    + (*rx)*(Kx[OPS_ACC1(1, 0)] + Kx[OPS_ACC1(0,0)]))*sd[OPS_ACC3(0,0)]
    - (*ry)*(Ky[OPS_ACC2(0, 1)] *sd[OPS_ACC3(0, 1)] + Ky[OPS_ACC2(0,0)]*sd[OPS_ACC3(0, -1)])
    - (*rx)*(Kx[OPS_ACC1(1, 0)] *sd[OPS_ACC3(1, 0)] + Kx[OPS_ACC1(0,0)]*sd[OPS_ACC3(-1, 0)]);
  rtemp[OPS_ACC0(0,0)] = rtemp[OPS_ACC0(0,0)] - smvp;
}

void tea_leaf_ppcg_inner2_kernel(double *sd, double *utemp, const double *z, const double *alpha, const double *beta) {
  sd[OPS_ACC0(0,0)] = (*alpha) * sd[OPS_ACC0(0,0)] + (*beta)*z[OPS_ACC2(0,0)];
  utemp[OPS_ACC1(0,0)] = utemp[OPS_ACC1(0,0)] + sd[OPS_ACC0(0,0)];
}

void tea_leaf_ppcg_reduce_kernel(const double *rstore, const double *r, const double *z, double *rnn) {
  *rnn = *rnn + (r[OPS_ACC1(0,0)] - rstore[OPS_ACC0(0,0)]) * z[OPS_ACC2(0,0)];
}

#endif
