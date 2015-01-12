#ifndef DRHOUUPDX_KERNEL_H
#define DRHOUUPDX_KERNEL_H

#include "vars.h"


void drhouupdx_kernel(const double *rhou_new, const double* rho_new, const double* rhoE_new, double *rhou_res) {
  
			double fni = rhou_new[OPS_ACC0(0)] * rhou_new[OPS_ACC0(0)] / rho_new[OPS_ACC1(0)] ;
			double p = gam1 * (rhoE_new[OPS_ACC2(0)] - 0.5 * fni);
			fni = fni + p;
			double fnim1 = rhou_new[OPS_ACC0(-1)] * rhou_new[OPS_ACC0(-1)] / rho_new[OPS_ACC1(-1)];
			p = gam1 * (rhoE_new[OPS_ACC2(-1)] - 0.5 * fnim1);
			fnim1 = fnim1 + p;
			double fnim2 = rhou_new[OPS_ACC0(-2)] * rhou_new[OPS_ACC0(-2)] / rho_new[OPS_ACC1(-2)];
			p = gam1 * (rhoE_new[OPS_ACC2(-2)] - 0.5 * fnim2);
			fnim2 = fnim2 + p;
			double fnip1 = rhou_new[OPS_ACC0(1)] * rhou_new[OPS_ACC0(1)] / rho_new[OPS_ACC1(1)];
			p = gam1 * (rhoE_new[OPS_ACC2(1)] - 0.5 * fnip1);
			fnip1 = fnip1 + p;
			double fnip2 = rhou_new[OPS_ACC0(2)] * rhou_new[OPS_ACC0(2)] / rho_new[OPS_ACC1(2)];
			p = gam1 * (rhoE_new[OPS_ACC2(2)] - 0.5 * fnip2);
			fnip2 = fnip2 + p;

			double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhou_res[OPS_ACC3(0)] = deriv;
}

#endif