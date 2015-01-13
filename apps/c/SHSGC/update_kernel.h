#ifndef UPDATE_KERNEL_H
#define UPDATE_KERNEL_H

#include "vars.h"


void update_kernel(double *rho_new, double *rhou_new, double *rhoE_new, const double *s) {
		rho_new[OPS_ACC0(0)]  = rho_new[OPS_ACC0(0)]  + s[OPS_ACC_MD3(0,0)];
		rhou_new[OPS_ACC1(0)] = rhou_new[OPS_ACC1(0)] + s[OPS_ACC_MD3(1,0)];
		rhoE_new[OPS_ACC2(0)] = rhoE_new[OPS_ACC2(0)] + s[OPS_ACC_MD3(2,0)];
}

#endif