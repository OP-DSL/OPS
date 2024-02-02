#ifndef DRHODX_KERNEL_H
#define DRHODX_KERNEL_H

#include "vars.h"


void drhoudx_kernel(const ACC<double> &rhou_new, ACC<double> &rho_res) {
        //double fni = rhou_new(0);
        double fnim1 = rhou_new(-1);
        double fnim2 = rhou_new(-2);
        double fnip1 = rhou_new(1);
        double fnip2 = rhou_new(2);

        double deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
        rho_res(0) = deriv;
}
#endif
