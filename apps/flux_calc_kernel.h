#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

void flux_calc_kernelx( double **vol_flux_x, double **xarea,
                        double **xvel0, double **xvel1) {

  //{0,0, 0,1};
  **vol_flux_x = 0.25 * dt * (**xarea) * ( (*xvel0[0]) + (*xvel0[1]) + (*xvel1[0]) + (*xvel1[1]) );

}

void flux_calc_kernely( double **vol_flux_y, double **yarea,
                        double **yvel0, double **yvel1) {

    //{0,0, 1,0};
  **vol_flux_y = 0.25 * dt * (**yarea) * ( (*yvel0[0]) + (*yvel0[1]) + (*yvel1[0]) + (*yvel1[1]) );

}
#endif
