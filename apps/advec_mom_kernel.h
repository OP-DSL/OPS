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

//vol_flux_x accessed with : {0,0, 0,1}

  **post_vol = **volume;
  **pre_vol  = **post_vol + (*vol_flux_x[1]) - (*vol_flux_x[0]);

}


void advec_mom_mass_flux_kernel( double **node_flux, double **mass_flux_x) {

  //mass_flux_x accessed with: {0,0, 1,0, 0,-1, 1,-1}

  **node_flux = 0.25 * ( (*mass_flux_x[2]) + (*mass_flux_x[0]) +
    (*mass_flux_x[3]) + (*mass_flux_x[1])); // Mass Flux

}

void advec_mom_post_advec_kernel( double **node_mass_post, double **post_vol,
                                  double **density1) {

  //post_vol accessed with: {0,0, -1,0, 0,-1, -1,-1}
  //density1 accessed with: {0,0, -1,0, 0,-1, -1,-1}

  **node_mass_post = 0.25 * ( (*density1[2]) * (*post_vol[2]) +
                              (*density1[0]) * (*post_vol[0]) +
                              (*density1[3]) * (*post_vol[3]) +
                              (*density1[1]) * (*post_vol[1]) );

}

void advec_mom_pre_advec_kernel( double **node_mass_pre, double **node_mass_post,
                                  double **node_flux) {

  //node_flux accessed with: {0,0, -1,0}
  **node_mass_pre = (**node_mass_post) - (*node_flux[1]) + (*node_flux[0]);

}


void advec_mom_kernel1( double **node_flux, double **node_mass_pre,
                        double **advec_vel, double **mom_flux,
                        double **celldx, double **vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 1,0}
  //celldx is accessed with {0,0, 1,0, -1,0, -2,0} striding in x
  //vel1 is accessed with {0,0, 1,0, 2,0, -1,0}

  double sigma, wind, width;
  double sigma2, wind2;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  double vdiffuw2, vdiffdw2, auw2, limiter2;

  sigma = fabs(**node_flux)/(*node_mass_pre[1]);
  sigma2 = fabs(**node_flux)/(*node_mass_pre[0]);

  width = **celldx;
  vdiffuw = (*vel1[1]) - (*vel1[2]);
  vdiffdw = (*vel1[0]) - (*vel1[1]);
  vdiffuw2 = (*vel1[0]) - (*vel1[3]);
  vdiffdw2 = -vdiffdw;

  auw = fabs(vdiffuw);
  adw = fabs(vdiffdw);
  auw2 = fabs(vdiffuw2);
  wind = 1.0;
  wind2 = 1.0;

  if(vdiffdw <= 0.0) wind = -1.0;
  if(vdiffdw2 <= 0.0) wind2 = -1.0;

  limiter = wind * MIN( width * ( (2.0 - sigma) * adw/width + (1.0 + sigma) *
                        auw/(*celldx[1]) )/6.0 , MIN(auw, adw) );
  limiter2= wind2* MIN( width * ( (2.0 - sigma2) * adw/width + (1.0 + sigma2) *
                        auw2/(*celldx[2]) )/6.0, MIN(auw2,adw) );

  if((vdiffuw * vdiffdw) <= 0.0) limiter = 0.0;
  if((vdiffuw2 * vdiffdw2) <= 0.0) limiter2 = 0.0;

  if( (**node_flux) < 0.0) {
    **advec_vel = (*vel1[1]) + (1.0 - sigma) * limiter;
  }
  else {
    **advec_vel = (*vel1[0]) + (1.0 - sigma2) * limiter2;
  }

  **mom_flux = (**advec_vel) * (**node_flux);

}

void advec_mom_kernel2( double **vel1, double **node_mass_post,
                        double **node_mass_pre, double **mom_flux) {

  //mom_flux accessed with: {0,0, -1,0}
  **vel1 = ( (**vel1) * (**node_mass_pre) + (*mom_flux[1]) - (*mom_flux[0]) ) / (**node_mass_post);
}






#endif
