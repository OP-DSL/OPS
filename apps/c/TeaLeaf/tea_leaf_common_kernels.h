#ifndef TEA_LEAF_COMMON_KERNEL_H
#define TEA_LEAF_COMMON_KERNEL_H

#include "data.h"
#include "definitions.h"

void tea_leaf_common_init_u_u0_kernel(double *u, double *u0, const double *energy, const double *density) {
	u [OPS_ACC0(0,0)]=energy[OPS_ACC2(0,0)]*density[OPS_ACC3(0,0)];
	u0[OPS_ACC1(0,0)]=energy[OPS_ACC2(0,0)]*density[OPS_ACC3(0,0)];
}

void tea_leaf_common_init_Kx_Ky_kernel(double *Kx, double *Ky, const double *w) {
	Kx[OPS_ACC0(0,0)]=(w[OPS_ACC2(-1,0 )]+w[OPS_ACC2(0,0)])/(2.0*w[OPS_ACC2(-1,0 )]*w[OPS_ACC2(0,0)]);
	Ky[OPS_ACC1(0,0)]=(w[OPS_ACC2( 0,-1)]+w[OPS_ACC2(0,0)])/(2.0*w[OPS_ACC2( 0,-1)]*w[OPS_ACC2(0,0)]);
}

void tea_leaf_common_init_kernel(double *w, double *r, const double *Kx, const double *Ky,
    const double *u,const double *rx,const double *ry) {
  w[OPS_ACC0(0,0)] = (1.0
        + (*ry)*(Ky[OPS_ACC3(0, 1)] + Ky[OPS_ACC3(0,0)])
        + (*rx)*(Kx[OPS_ACC2(1, 0)] + Kx[OPS_ACC2(0,0)]))*u[OPS_ACC4(0,0)]
        - (*ry)*(Ky[OPS_ACC3(0, 1)] *u[OPS_ACC4(0, 1)] + Ky[OPS_ACC3(0,0)]*u[OPS_ACC4(0, -1)])
        - (*rx)*(Kx[OPS_ACC2(1, 0)] *u[OPS_ACC4(1, 0)] + Kx[OPS_ACC2(0,0)]*u[OPS_ACC4(-1, 0)]);
    r[OPS_ACC1(0,0)] = u[OPS_ACC4(0,0)] - w[OPS_ACC0(0,0)];
}

void tea_leaf_common_residual_kernel(double *r, const double *Kx, const double *Ky,
    const double *u,const double *u0,const double *rx,const double *ry) {
	double smvp = 0.0;
  smvp = (1.0
        + (*ry)*(Ky[OPS_ACC2(0, 1)] + Ky[OPS_ACC2(0,0)])
        + (*rx)*(Kx[OPS_ACC1(1, 0)] + Kx[OPS_ACC1(0,0)]))*u[OPS_ACC3(0,0)]
        - (*ry)*(Ky[OPS_ACC2(0, 1)] *u[OPS_ACC3(0, 1)] + Ky[OPS_ACC2(0,0)]*u[OPS_ACC3(0, -1)])
        - (*rx)*(Kx[OPS_ACC1(1, 0)] *u[OPS_ACC3(1, 0)] + Kx[OPS_ACC1(0,0)]*u[OPS_ACC3(-1, 0)]);
    r[OPS_ACC0(0,0)] = u0[OPS_ACC4(0,0)] - smvp;
}

void tea_leaf_common_init_diag_init_kernel(double *Mi, const double *Kx, const double *Ky,
	const double *rx, const double *ry) {
	Mi[OPS_ACC0(0,0)] = 1.0/(1.0
			+(*ry)*(Ky[OPS_ACC2(0,1)] + Ky[OPS_ACC2(0,0)])
			+(*rx)*(Kx[OPS_ACC1(1,0)] + Kx[OPS_ACC1(0,0)]));
}
#endif
