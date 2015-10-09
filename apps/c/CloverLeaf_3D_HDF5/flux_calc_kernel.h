#ifndef FLUX_CALC_KERNEL_H
#define FLUX_CALC_KERNEL_H

void flux_calc_kernelx( double *vol_flux_x, const double *xarea,
                        const double *xvel0, const double *xvel1) {

  vol_flux_x[OPS_ACC0(0,0,0)] = 0.125 * dt * (xarea[OPS_ACC1(0,0,0)]) *
  ( xvel0[OPS_ACC2(0,0,0)] + xvel0[OPS_ACC2(0,1,0)] + xvel0[OPS_ACC2(0,0,1)] + xvel0[OPS_ACC2(0,1,1)] +
    xvel1[OPS_ACC3(0,0,0)] + xvel1[OPS_ACC3(0,1,0)] + xvel1[OPS_ACC3(0,0,1)] + xvel1[OPS_ACC3(0,1,1)]);
}

void flux_calc_kernely( double *vol_flux_y, const double *yarea,
                        const double *yvel0, const double *yvel1) {

  vol_flux_y[OPS_ACC0(0,0,0)] = 0.125 * dt * (yarea[OPS_ACC1(0,0,0)]) *
  ( yvel0[OPS_ACC2(0,0,0)] + yvel0[OPS_ACC2(1,0,0)] + yvel0[OPS_ACC2(0,0,1)] + yvel0[OPS_ACC2(1,0,1)] +
    yvel1[OPS_ACC3(0,0,0)] + yvel1[OPS_ACC3(1,0,0)] + yvel1[OPS_ACC3(0,0,1)] + yvel1[OPS_ACC3(1,0,1)]);
}

void flux_calc_kernelz( double *vol_flux_z, const double *zarea,
                        const double *zvel0, const double *zvel1) {

  vol_flux_z[OPS_ACC0(0,0,0)] = 0.125 * dt * (zarea[OPS_ACC1(0,0,0)]) *
  ( zvel0[OPS_ACC2(0,0,0)] + zvel0[OPS_ACC2(1,0,0)] + zvel0[OPS_ACC2(1,0,0)] + zvel0[OPS_ACC2(1,1,0)] +
    zvel1[OPS_ACC3(0,0,0)] + zvel1[OPS_ACC3(1,0,0)] + zvel1[OPS_ACC3(0,1,0)] + zvel1[OPS_ACC3(1,1,0)]);
}


#endif