#ifndef SAVE_KERNEL_H
#define SAVE_KERNEL_H

#include "vars.h"


void save_kernel(ACC<double> &rho_old, ACC<double> &rhou_old, ACC<double> &rhoE_old,
                       const ACC<double> &rho_new, const ACC<double> &rhou_new, const ACC<double> &rhoE_new) {
      rho_old(0)=rho_new(0);
      rhou_old(0)=rhou_new(0);
      rhoE_old(0)=rhoE_new(0);
}
#endif
