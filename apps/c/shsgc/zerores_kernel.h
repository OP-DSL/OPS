#ifndef ZERORES_KERNEL_H
#define ZERORES_KERNEL_H

#include "vars.h"


void zerores_kernel(ACC<double> &rho_res, ACC<double> &rhou_res, ACC<double> &rhoE_res) {
      rho_res(0) = 0.0;
      rhou_res(0) = 0.0;
      rhoE_res(0) = 0.0;
}
#endif
