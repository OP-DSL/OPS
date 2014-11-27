#ifndef SAVE_KERNEL_H
#define SAVE_KERNEL_H

#include "vars.h"


void save_kernel(double *rho_old, double *rhou_old, double *rhoE_old,
                       const double *rho_new, const double *rhou_new, const double *rhoE_new) {
      rho_old[0]=rho_new[0];
      rhou_old[0]=rhou_new[0];
      rhoE_old[0]=rhoE_new[0];
}

#endif