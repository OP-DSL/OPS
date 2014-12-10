#ifndef XDER1_KERNEL_H
#define XDER1_KERNEL_H

#include "vars.h"


void xder1_kernel(double *fn, double *dfn) {
        double fni = fn[OPS_ACC0(0)];
        double fnim1 = fn[OPS_ACC0(-1)];
        double fnim2 = fn[OPS_ACC0(-2)];
        double fnip1 = fn[OPS_ACC0(1)];
        double fnip2 = fn[OPS_ACC0(2)];

        dfn [OPS_ACC1(0)] = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
        
}

#endif



