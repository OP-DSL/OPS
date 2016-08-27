#ifndef TEA_LEAF_CG_KERNEL_H
#define TEA_LEAF_CG_KERNEL_H

#include "data.h"
#include "definitions.h"


void tea_leaf_cg_calc_w_reduce_kernel(double *w, const double *Kx, const double *Ky, const double *p, const double *rx, const double *ry, double *pw) {
  w[OPS_ACC0(0,0)] = (1.0
                + (*ry)*(Ky[OPS_ACC2(0,1)] + Ky[OPS_ACC2(0,0)])                      
                + (*rx)*(Ky[OPS_ACC1(1,0)] + Kx[OPS_ACC1(0,0)]))*p[OPS_ACC3(0,0)]             
                - (*ry)*(Ky[OPS_ACC2(0,1)]*p[OPS_ACC3(0,1)] + Ky[OPS_ACC2(0,0)]*p[OPS_ACC3(0,-1)])  
                - (*rx)*(Ky[OPS_ACC1(1,0)]*p[OPS_ACC3(1,0)] + Kx[OPS_ACC1(0,0)]*p[OPS_ACC3(-1,0)]);
  *pw = *pw + w[OPS_ACC0(0,0)]*p[OPS_ACC3(0,0)];
}

void tea_leaf_cg_calc_ur_r_reduce_kernel(double * r, const double * w, const double * alpha, double *rnn) {
  r[OPS_ACC0(0,0)] = r[OPS_ACC0(0,0)] - (*alpha)*w[OPS_ACC1(0,0)];
  *rnn = *rnn +  r[OPS_ACC0(0,0)]*r[OPS_ACC0(0,0)];
}

#endif
