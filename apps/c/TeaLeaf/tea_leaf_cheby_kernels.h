#ifndef TEA_LEAF_CHEBY_KERNEL_H
#define TEA_LEAF_CHEBY_KERNEL_H

#include "data.h"
#include "definitions.h"

void tea_leaf_cheby_init_kernel(double *w, double *r, const double *Kx, const double *Ky,
		const double *u,const double *u0,const double *rx,const double *ry) {
	w[OPS_ACC0(0,0)] = (1.0
        + (*ry)*(Ky[OPS_ACC3(0, 1)] + Ky[OPS_ACC3(0,0)])
        + (*rx)*(Kx[OPS_ACC2(1, 0)] + Kx[OPS_ACC2(0,0)]))*u[OPS_ACC4(0,0)]
        - (*ry)*(Ky[OPS_ACC3(0, 1)] *u[OPS_ACC4(0, 1)] + Ky[OPS_ACC3(0,0)]*u[OPS_ACC4(0, -1)])
        - (*rx)*(Kx[OPS_ACC2(1, 0)] *u[OPS_ACC4(1, 0)] + Kx[OPS_ACC2(0,0)]*u[OPS_ACC4(-1, 0)]);
    r[OPS_ACC1(0,0)] = u0[OPS_ACC5(0,0)] - w[OPS_ACC0(0,0)];
}

#endif
