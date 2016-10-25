#ifndef TEA_LEAF_CHEBY_KERNEL_H
#define TEA_LEAF_CHEBY_KERNEL_H

#include "data.h"
#include "definitions.h"

void tea_leaf_jacobi_kernel(double *u1, const double *Kx, const double *Ky,
		const double *un,const double *u0,const double *rx,const double *ry, double *error) {
	u1[OPS_ACC0(0,0)] = (u0[OPS_ACC4(0,0)] 
		+ (*rx)*(Kx[OPS_ACC1(1, 0)] *un[OPS_ACC3(1, 0)] + Kx[OPS_ACC1(0,0)]*un[OPS_ACC3(-1, 0)])
		+ (*ry)*(Ky[OPS_ACC2(0, 1)] *un[OPS_ACC3(0, 1)] + Ky[OPS_ACC2(0,0)]*un[OPS_ACC3(0, -1)]))
			/(1.0
				+ (*rx)*(Kx[OPS_ACC1(1, 0)] + Kx[OPS_ACC1(0,0)])
				+ (*ry)*(Ky[OPS_ACC2(0, 1)] + Ky[OPS_ACC2(0,0)]));

    *error = *error + fabs(u1[OPS_ACC0(0,0)] - un[OPS_ACC3(0,0)]);
}

#endif
