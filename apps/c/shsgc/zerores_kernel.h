#ifndef ZERORES_KERNEL_H
#define ZERORES_KERNEL_H

#include "vars.h"


void zerores_kernel(double *rho_res, double *rhou_res, double *rhoE_res) {
      rho_res[OPS_ACC0(0)] = 0.0;
      rhou_res[OPS_ACC1(0)] = 0.0;
      rhoE_res[OPS_ACC2(0)] = 0.0;
}
#endif
