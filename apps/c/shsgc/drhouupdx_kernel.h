#ifndef DRHOUUPDX_KERNEL_H
#define DRHOUUPDX_KERNEL_H

#include "vars.h"


void drhouupdx_kernel(const ACC<double> &rhou_new, const ACC<double> &rho_new, const ACC<double> &rhoE_new, ACC<double> &rhou_res) {

			double fni = rhou_new(0) * rhou_new(0) / rho_new(0) ;
			double p = gam1 * (rhoE_new(0) - 0.5 * fni);
			fni = fni + p;
			double fnim1 = rhou_new(-1) * rhou_new(-1) / rho_new(-1);
			p = gam1 * (rhoE_new(-1) - 0.5 * fnim1);
			fnim1 = fnim1 + p;
			double fnim2 = rhou_new(-2) * rhou_new(-2) / rho_new(-2);
			p = gam1 * (rhoE_new(-2) - 0.5 * fnim2);
			fnim2 = fnim2 + p;
			double fnip1 = rhou_new(1) * rhou_new(1) / rho_new(1);
			p = gam1 * (rhoE_new(1) - 0.5 * fnip1);
			fnip1 = fnip1 + p;
			double fnip2 = rhou_new(2) * rhou_new(2) / rho_new(2);
			p = gam1 * (rhoE_new(2) - 0.5 * fnip2);
			fnip2 = fnip2 + p;

			double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
			rhou_res(0) = deriv;
}
#endif
