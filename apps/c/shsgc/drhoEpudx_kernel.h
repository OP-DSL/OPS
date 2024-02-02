#ifndef DRHOEPUDX_KERNEL_H
#define DRHOEPUDX_KERNEL_H

#include "vars.h"


void drhoEpudx_kernel(const ACC<double> &rhou_new, const ACC<double>& rho_new, const ACC<double>& rhoE_new, ACC<double> &rhoE_res) {

			double fni = rhou_new(0) * rhou_new(0) / rho_new(0) ;
			double p = gam1 * (rhoE_new(0) - 0.5 * fni);
			fni = (rhoE_new(0) + p) * rhou_new(0) / rho_new(0) ;

			double fnim1 = rhou_new(-1) * rhou_new(-1) / rho_new(-1);
			p = gam1 * (rhoE_new(-1) - 0.5 * fnim1);
			fnim1 = (rhoE_new(-1) + p) * rhou_new(-1) / rho_new(-1);

			double fnim2 = rhou_new(-2) * rhou_new(-2) / rho_new(-2);
			p = gam1 * (rhoE_new(-2) - 0.5 * fnim2);
			fnim2 = (rhoE_new(-2) + p ) * rhou_new(-2) / rho_new(-2);

			double fnip1 = rhou_new(1) * rhou_new(1) / rho_new(1);
			p = gam1 * (rhoE_new(1) - 0.5 * fnip1);
			fnip1 = (rhoE_new(1) + p) * rhou_new(1) / rho_new(1);

			double fnip2 = rhou_new(2) * rhou_new(2) / rho_new(2);
			p = gam1 * (rhoE_new(2) - 0.5 * fnip2);
			fnip2 = (rhoE_new(2) + p) * rhou_new(2) / rho_new(2);

			double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhoE_res(0) = deriv;
}
#endif
