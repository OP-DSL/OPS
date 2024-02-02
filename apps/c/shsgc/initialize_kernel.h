#ifndef INITIALIZE_KERNEL_H
#define INITIALIZE_KERNEL_H

#include "vars.h"


void initialize_kernel(ACC<double> &x,ACC<double> &rho_new, ACC<double> &rhou_new, ACC<double> &rhoE_new,
                       ACC<double>& rhoin, int *idx) {
  x(0) = xmin + (idx[0]-2) * dx;
  if (x(0) >= -4.0){
		rho_new(0) = 1.0 + eps * sin(lambda *x(0));
		rhou_new(0) = ur * rho_new(0);
		rhoE_new(0) = (pr / gam1) + 0.5 * pow(rhou_new(0),2)/rho_new(0);
	}
	else {
		rho_new(0) = rhol;
		rhou_new(0) = ul2 * rho_new(0);
		rhoE_new(0) = (pl / gam1) + 0.5 * pow(rhou_new(0),2)/rho_new(0);
	}

	rhoin(0) = gam1 * (rhoE_new(0) - 0.5 * rhou_new(0) * rhou_new(0) / rho_new(0));

}
#endif
