#ifndef SAVE_KERNEL_H
#define SAVE_KERNEL_H

#include "vars.h"


void save_kernel(double *rho_old, double *rhou_old, double *rhoE_old,
                       const double *rho_new, const double *rhou_new, const double *rhoE_new) {
      rho_old[OPS_ACC0(0)]=rho_new[OPS_ACC3(0)];
      rhou_old[OPS_ACC1(0)]=rhou_new[OPS_ACC4(0)];
      rhoE_old[OPS_ACC2(0)]=rhoE_new[OPS_ACC5(0)];
}

#endif