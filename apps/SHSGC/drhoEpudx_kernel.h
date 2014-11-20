#ifndef DRHOEPUDX_KERNEL_H
#define DRHOEPUDX_KERNEL_H

#include "vars.h"


void drhoEpudx_kernel(double *rhou_new, double* rho_new, double* rhoE_new, double *rhoE_res) {
  			
			double fni = rhou_new[OPS_ACC0(0)] * rhou_new[OPS_ACC0(0)] / rho_new[OPS_ACC1(0)] ;
			double p = gam1 * (rhoE_new[OPS_ACC2(0)] - 0.5 * fni);
			fni = (rhoE_new[OPS_ACC2(0)] + p) * rhou_new[OPS_ACC0(0)] / rho_new[OPS_ACC1(0)] ;
			
			double fnim1 = rhou_new[OPS_ACC0(-1)] * rhou_new[OPS_ACC0(-1)] / rho_new[OPS_ACC1(-1)];
			p = gam1 * (rhoE_new[OPS_ACC2(-1)] - 0.5 * fnim1);
			fnim1 = (rhoE_new[OPS_ACC2(-1)] + p) * rhou_new[OPS_ACC0(-1)] / rho_new[OPS_ACC1(-1)];
			
			double fnim2 = rhou_new[OPS_ACC0(-2)] * rhou_new[OPS_ACC0(-2)] / rho_new[OPS_ACC1(-2)];
			p = gam1 * (rhoE_new[OPS_ACC2(-2)] - 0.5 * fnim2);
			fnim2 = (rhoE_new[OPS_ACC2(-2)] + p ) * rhou_new[OPS_ACC0(-2)] / rho_new[OPS_ACC1(-2)];
			
			double fnip1 = rhou_new[OPS_ACC0(1)] * rhou_new[OPS_ACC0(1)] / rho_new[OPS_ACC1(1)];
			p = gam1 * (rhoE_new[OPS_ACC2(1)] - 0.5 * fnip1);
			fnip1 = (rhoE_new[OPS_ACC2(1)] + p) * rhou_new[OPS_ACC0(1)] / rho_new[OPS_ACC1(1)];
			
			double fnip2 = rhou_new[OPS_ACC0(2)] * rhou_new[OPS_ACC0(2)] / rho_new[OPS_ACC1(2)];
			p = gam1 * (rhoE_new[OPS_ACC2(2)] - 0.5 * fnip2);
			fnip2 = (rhoE_new[OPS_ACC2(2)] + p) * rhou_new[OPS_ACC0(2)] / rho_new[OPS_ACC1(2)];

			double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhoE_res[OPS_ACC3(0)] = deriv;			
}

#endif