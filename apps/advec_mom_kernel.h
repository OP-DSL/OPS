#ifndef ADVEC_MOM_KERNEL_H
#define ADVEC_MOM_KERNEL_H

void advec_mom_x1_kernel( double **pre_vol, double **post_vol,
                          double **volume,
                          double **vol_flux_x, double **vol_flux_y) {

  //vol_flux_y accessed with : 0,0, 1,0
  //vol_flux_y accessed with : 0,0, 0,1

  **post_vol = **volume + (*vol_flux_y[1]) - (*vol_flux_y[0]);
  **pre_vol = **post_vol + (*vol_flux_x[1]) - (*vol_flux_x[0]);

}

void advec_mom_y1_kernel( double **pre_vol, double **post_vol,
                          double **volume,
                          double **vol_flux_x, double **vol_flux_y) {

  //vol_flux_y accessed with : 0,0, 1,0
  //vol_flux_y accessed with : 0,0, 0,1

  **post_vol = **volume + (*vol_flux_x[1]) - (*vol_flux_x[0]);
  **pre_vol  = **post_vol + (*vol_flux_y[1]) - (*vol_flux_y[0]);

}

void advec_mom_x2_kernel( double **pre_vol, double **post_vol,
                          double **volume,
                          double **vol_flux_y) {

  //vol_flux_y accessed with : 0,0, 0,1

  **post_vol = **volume;
  **pre_vol  = **post_vol + (*vol_flux_y[1]) - (*vol_flux_y[0]);

}

void advec_mom_y2_kernel( double **pre_vol, double **post_vol,
                          double **volume,
                          double **vol_flux_x) {

  //vol_flux_x accessed with : 0,0, 0,1

  **post_vol = **volume;
  **pre_vol  = **post_vol + (*vol_flux_x[1]) - (*vol_flux_x[0]);

}

#endif
