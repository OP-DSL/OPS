#ifndef XDER12_KERNEL_H
#define XDER12_KERNEL_H

#include "vars.h"


void xder12_kernel(double *fn, double *dfn1, double *dfn2) {
        double fni = fn[OPS_ACC0(0)];
        double fnim1 = fn[OPS_ACC0(-1)];
        double fnim2 = fn[OPS_ACC0(-2)];
        double fnip1 = fn[OPS_ACC0(1)];
        double fnip2 = fn[OPS_ACC0(2)];

        dfn1[OPS_ACC1(0)] = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.0 * dx);
        dfn2[OPS_ACC2(0)] = (-fnim2 + fnip2 + 16.0* (fnip1 + fnim1) -30.0 * fni)/(12.0 * pow(dx,2));
}

#endif


