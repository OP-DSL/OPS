#ifndef INITIALIZE_KERNEL_H
#define INITIALIZE_KERNEL_H

#include "vars.h"


void initialize_kernel(double *x,double *rho_new, double *rhou_new, double *rhoE_new, 
                       double* rhoin, int *idx) {
  x[OPS_ACC0(0)] = xmin + (idx[0]-2) * dx;  
  if (x[OPS_ACC0(0)] >= -4.0){
		rho_new[OPS_ACC1(0)] = 1.0 + eps * sin(lambda *x[OPS_ACC0(0)]);
		rhou_new[OPS_ACC2(0)] = ur * rho_new[OPS_ACC1(0)];
		rhoE_new[OPS_ACC3(0)] = (pr / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
	}
	else {
		rho_new[OPS_ACC1(0)] = rhol;
		rhou_new[OPS_ACC2(0)] = ul * rho_new[OPS_ACC1(0)];
		rhoE_new[OPS_ACC3(0)] = (pl / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
	}
	
	rhoin[OPS_ACC4(0)] = gam1 * (rhoE_new[OPS_ACC3(0)] - 0.5 * rhou_new[OPS_ACC2(0)] * rhou_new[OPS_ACC2(0)] / rho_new[OPS_ACC1(0)]);
    
}

#endif