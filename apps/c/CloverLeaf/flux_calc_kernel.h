#ifndef FLUX_CALC_KERNEL_H
#define FLUX_CALC_KERNEL_H

void flux_calc_kernelx( ACC<double> &vol_flux_x, const ACC<double> &xarea,
                        const ACC<double> &xvel0, const ACC<double> &xvel1) {

  //{0,0, 0,1};
  vol_flux_x(0,0) = 0.25 * dt * (xarea(0,0)) *
  ( (xvel0(0,0)) + (xvel0(0,1)) + (xvel1(0,0)) + (xvel1(0,1)) );

}

void flux_calc_kernely( ACC<double> &vol_flux_y, const ACC<double> &yarea,
                        const ACC<double> &yvel0, const ACC<double> &yvel1) {

    //{0,0, 1,0};
  vol_flux_y(0,0) = 0.25 * dt * (yarea(0,0)) *
  ( (yvel0(0,0)) + (yvel0(1,0)) + (yvel1(0,0)) + (yvel1(1,0)) );

}
#endif
