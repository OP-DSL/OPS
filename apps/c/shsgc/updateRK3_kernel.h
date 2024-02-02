#ifndef UPDATERK3_KERNEL_H
#define UPDATERK3_KERNEL_H

#include "vars.h"


void updateRK3_kernel(ACC<double> &rho_new, ACC<double>& rhou_new, ACC<double>& rhoE_new,
                      ACC<double> &rho_old, ACC<double>& rhou_old, ACC<double>& rhoE_old,
                      const ACC<double> &rho_res, const ACC<double> &rhou_res, const ACC<double> &rhoE_res,
                      const double* a1, const double* a2) {

			rho_new(0) = rho_old(0) + dt * a1[0] * (-rho_res(0));
			rhou_new(0) = rhou_old(0) + dt * a1[0] * (-rhou_res(0));
			rhoE_new(0) = rhoE_old(0) + dt * a1[0] * (-rhoE_res(0));
			// update old state
			rho_old(0) = rho_old(0) + dt * a2[0] * (-rho_res(0));
			rhou_old(0) = rhou_old(0) + dt * a2[0] * (-rhou_res(0));
			rhoE_old(0) = rhoE_old(0) + dt * a2[0] * (-rhoE_res(0));
}
#endif
