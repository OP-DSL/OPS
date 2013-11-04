#ifndef FLUX_CALC_KERNEL_H
#define FLUX_CALC_KERNEL_H

void flux_calc_kernelx( double *vol_flux_x, double *xarea,
                        double *xvel0, double *xvel1) {

  //{0,0, 0,1};
  vol_flux_x[OPS_ACC0(0,0)] = 0.25 * dt * (xarea[OPS_ACC1(0,0)]) *
  ( (xvel0[OPS_ACC2(0,0)]) + (xvel0[OPS_ACC2(0,1)]) + (xvel1[OPS_ACC3(0,0)]) + (xvel1[OPS_ACC3(0,1)]) );

}

void flux_calc_kernely( double *vol_flux_y, double *yarea,
                        double *yvel0, double *yvel1) {

    //{0,0, 1,0};
  vol_flux_y[OPS_ACC0(0,0)] = 0.25 * dt * (yarea[OPS_ACC1(0,0)]) *
  ( (yvel0[OPS_ACC2(0,0)]) + (yvel0[OPS_ACC2(1,0)]) + (yvel1[OPS_ACC3(0,0)]) + (yvel1[OPS_ACC3(1,0)]) );

}
#endif
