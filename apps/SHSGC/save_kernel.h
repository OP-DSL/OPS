#ifndef SAVE_KERNEL_H
#define SAVE_KERNEL_H

#include "vars.h"


void save_kernel(double *rho_old, double *rhou_old, double *rhoE_old,
                       double *rho_new, double *rhou_new, double *rhoE_new) {
      rho_old[0]=rho_new[0];
      rhou_old[0]=rhou_new[0];
      rhoE_old[0]=rhoE_new[0];
}

#endif