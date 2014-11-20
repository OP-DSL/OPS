#ifndef DRHODX_KERNEL_H
#define DRHODX_KERNEL_H

#include "vars.h"


void drhoudx_kernel(double *rhou_new, double *rho_res) {
        double fni = rhou_new[OPS_ACC0(0)];
        double fnim1 = rhou_new[OPS_ACC0(-1)];
        double fnim2 = rhou_new[OPS_ACC0(-2)];
        double fnip1 = rhou_new[OPS_ACC0(1)];
        double fnip2 = rhou_new[OPS_ACC0(2)];

        double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
        rho_res[OPS_ACC1(0)] = deriv;
}

#endif