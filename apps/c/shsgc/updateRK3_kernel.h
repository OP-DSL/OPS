#ifndef UPDATERK3_KERNEL_H
#define UPDATERK3_KERNEL_H

#include "vars.h"


void updateRK3_kernel(double *rho_new, double* rhou_new, double* rhoE_new,
                      double *rho_old, double* rhou_old, double* rhoE_old,
                      const double *rho_res, const double *rhou_res, const double *rhoE_res,
                      const double* a1, const double* a2) {

			rho_new[OPS_ACC0(0)] = rho_old[OPS_ACC3(0)] + dt * a1[0] * (-rho_res[OPS_ACC6(0)]);
			rhou_new[OPS_ACC1(0)] = rhou_old[OPS_ACC4(0)] + dt * a1[0] * (-rhou_res[OPS_ACC7(0)]);
			rhoE_new[OPS_ACC2(0)] = rhoE_old[OPS_ACC5(0)] + dt * a1[0] * (-rhoE_res[OPS_ACC8(0)]);
			// update old state
			rho_old[OPS_ACC3(0)] = rho_old[OPS_ACC3(0)] + dt * a2[0] * (-rho_res[OPS_ACC6(0)]);
			rhou_old[OPS_ACC4(0)] = rhou_old[OPS_ACC4(0)] + dt * a2[0] * (-rhou_res[OPS_ACC7(0)]);
			rhoE_old[OPS_ACC5(0)] = rhoE_old[OPS_ACC5(0)] + dt * a2[0] * (-rhoE_res[OPS_ACC8(0)]);
}
#endif
