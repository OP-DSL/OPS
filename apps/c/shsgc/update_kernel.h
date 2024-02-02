#ifndef UPDATE_KERNEL_H
#define UPDATE_KERNEL_H

#include "vars.h"


void update_kernel(ACC<double> &rho_new, ACC<double> &rhou_new, ACC<double> &rhoE_new, const ACC<double> &s) {
		rho_new(0)  = rho_new(0)  + s(0,0);
		rhou_new(0) = rhou_new(0) + s(1,0);
		rhoE_new(0) = rhoE_new(0) + s(2,0);
}
#endif
