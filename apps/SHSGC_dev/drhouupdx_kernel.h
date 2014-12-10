#ifndef DRHOUUPDX_KERNEL_H
#define DRHOUUPDX_KERNEL_H

#include "vars.h"


void drhouupdx_kernel(double *rhou_new, double* rho_new, double* rhoE_new, double *fn) {
  
			double fni = rhou_new[OPS_ACC0(0)] * rhou_new[OPS_ACC0(0)] / rho_new[OPS_ACC1(0)] ;
			double p = gam1 * (rhoE_new[OPS_ACC2(0)] - 0.5 * fni);
			fni = fni + p;
			fn[OPS_ACC3(0)] = fni;
}

#endif
