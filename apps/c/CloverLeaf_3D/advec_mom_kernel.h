#ifndef ADVEC_MOM_KERNEL_H
#define ADVEC_MOM_KERNEL_H

inline void advec_mom_kernel_x1( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_x, const double *vol_flux_y, const double *vol_flux_z) {

  post_vol[OPS_ACC1(0,0,0)] = volume[OPS_ACC2(0,0,0)] + vol_flux_y[OPS_ACC4(0,1,0)] -  vol_flux_y[OPS_ACC4(0,0,0)]
                                                      + vol_flux_z[OPS_ACC5(0,0,1)] -  vol_flux_z[OPS_ACC5(0,0,0)];
  pre_vol[OPS_ACC0(0,0,0)] = post_vol[OPS_ACC1(0,0,0)] + vol_flux_x[OPS_ACC3(1,0,0)] - vol_flux_x[OPS_ACC3(0,0,0)];

}

inline void advec_mom_kernel_z1( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_x, const double *vol_flux_y, const double *vol_flux_z) {

  post_vol[OPS_ACC1(0,0,0)] = volume[OPS_ACC2(0,0,0)] + vol_flux_x[OPS_ACC3(1,0,0)] -  vol_flux_x[OPS_ACC3(0,0,0)]
                                                      + vol_flux_y[OPS_ACC4(0,1,0)] -  vol_flux_y[OPS_ACC4(0,0,0)];
  pre_vol[OPS_ACC0(0,0,0)] = post_vol[OPS_ACC1(0,0,0)] + vol_flux_z[OPS_ACC5(0,0,1)] - vol_flux_z[OPS_ACC5(0,0,0)];

}

inline void advec_mom_kernel_x2( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_y,const double *vol_flux_z) {

  post_vol[OPS_ACC1(0,0,0)]  = volume[OPS_ACC2(0,0,0)]  + vol_flux_z[OPS_ACC4(0,0,1)] - vol_flux_z[OPS_ACC4(0,0,0)];
  pre_vol[OPS_ACC0(0,0,0)]   = post_vol[OPS_ACC1(0,0,0)]  + vol_flux_y[OPS_ACC3(0,1,0)] - vol_flux_y[OPS_ACC3(0,0,0)];

}

inline void advec_mom_kernel_y2( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_x,const double *vol_flux_y) {

  post_vol[OPS_ACC1(0,0,0)]  = volume[OPS_ACC2(0,0,0)]  + vol_flux_x[OPS_ACC3(1,0,0)] - vol_flux_x[OPS_ACC3(0,0,0)] ;
  pre_vol[OPS_ACC0(0,0,0)]   = post_vol[OPS_ACC1(0,0,0)]  + vol_flux_y[OPS_ACC4(0,1,0)] - vol_flux_y[OPS_ACC4(0,0,0)];

}

inline void advec_mom_kernel_x3( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_x) {

  post_vol[OPS_ACC1(0,0,0)]  = volume[OPS_ACC2(0,0,0)];
  pre_vol[OPS_ACC0(0,0,0)]   = post_vol[OPS_ACC1(0,0,0)]  + vol_flux_x[OPS_ACC3(1,0,0)] - vol_flux_x[OPS_ACC3(0,0,0)];

}

inline void advec_mom_kernel_z3( double *pre_vol, double *post_vol,
                          const double *volume,
                          const double *vol_flux_z) {

  post_vol[OPS_ACC1(0,0,0)]  = volume[OPS_ACC2(0,0,0)];
  pre_vol[OPS_ACC0(0,0,0)]   = post_vol[OPS_ACC1(0,0,0)]  + vol_flux_z[OPS_ACC3(0,0,1)] - vol_flux_z[OPS_ACC3(0,0,0)];

}


////////////////////////



inline void advec_mom_kernel_mass_flux_x( double *node_flux, const double *mass_flux_x) {

  //mass_flux_x accessed with: {0,0, 1,0, 0,-1, 1,-1}

  node_flux[OPS_ACC0(0,0,0)] = 0.125 * ( mass_flux_x[OPS_ACC1(0,-1,0)] + mass_flux_x[OPS_ACC1(0,0,0)] +
                                         mass_flux_x[OPS_ACC1(1,-1,0)] + mass_flux_x[OPS_ACC1(1,0,0)] +
                                         mass_flux_x[OPS_ACC1(0,-1,-1)] + mass_flux_x[OPS_ACC1(0,0,-1)] +
                                         mass_flux_x[OPS_ACC1(1,-1,-1)] + mass_flux_x[OPS_ACC1(1,0,-1)] ); // Mass Flux in x
}


inline void advec_mom_kernel_mass_flux_y( double *node_flux, const double *mass_flux_y) {

  //mass_flux_y accessed with: {0,0, 0,1, -1,0, -1,1}

  node_flux[OPS_ACC0(0,0,0)] = 0.125 * ( mass_flux_y[OPS_ACC1(-1,0,0)] + mass_flux_y[OPS_ACC1(0,0,0)] +
                                         mass_flux_y[OPS_ACC1(-1,1,0)] + mass_flux_y[OPS_ACC1(0,1,0)] +
                                         mass_flux_y[OPS_ACC1(-1,0,-1)] + mass_flux_y[OPS_ACC1(0,0,-1)] +
                                         mass_flux_y[OPS_ACC1(-1,1,-1)] + mass_flux_y[OPS_ACC1(0,1,-1)] ); // Mass Flux in y
}

inline void advec_mom_kernel_mass_flux_z( double *node_flux, const double *mass_flux_z) {

  //mass_flux_y accessed with: {0,0, 0,1, -1,0, -1,1}

  node_flux[OPS_ACC0(0,0,0)] = 0.125 * ( mass_flux_z[OPS_ACC1(-1,0,0)] + mass_flux_z[OPS_ACC1(0,0,0)] +
                                         mass_flux_z[OPS_ACC1(-1,0,1)] + mass_flux_z[OPS_ACC1(0,0,1)] +
                                         mass_flux_z[OPS_ACC1(-1,-1,0)] + mass_flux_z[OPS_ACC1(0,-1,0)] +
                                         mass_flux_z[OPS_ACC1(-1,-1,1)] + mass_flux_z[OPS_ACC1(0,-1,1)] ); // Mass Flux in z
}


inline void advec_mom_kernel_post_pre_advec_x( double *node_mass_post, const double *post_vol,
                                  const double *density1, double *node_mass_pre, const double *node_flux) {

  node_mass_post[OPS_ACC0(0,0,0)] = 0.125 * ( density1[OPS_ACC2(0,-1,0)] * post_vol[OPS_ACC1(0,-1,0)] +
                                              density1[OPS_ACC2(0,0,0)]   * post_vol[OPS_ACC1(0,0,0)]   +
                                              density1[OPS_ACC2(-1,-1,0)] * post_vol[OPS_ACC1(-1,-1,0)] +
                                              density1[OPS_ACC2(-1,0,0)]  * post_vol[OPS_ACC1(-1,0,0)] +
                                              density1[OPS_ACC2(0,-1,-1)] * post_vol[OPS_ACC1(0,-1,-1)] +
                                              density1[OPS_ACC2(0,0,-1)]   * post_vol[OPS_ACC1(0,0,-1)]   +
                                              density1[OPS_ACC2(-1,-1,-1)] * post_vol[OPS_ACC1(-1,-1,-1)] +
                                              density1[OPS_ACC2(-1,0,-1)]  * post_vol[OPS_ACC1(-1,0,-1)]  );

  node_mass_pre[OPS_ACC3(0,0,0)] = node_mass_post[OPS_ACC0(0,0,0)] - node_flux[OPS_ACC4(-1,0,0)] + node_flux[OPS_ACC4(0,0,0)];

}

inline void advec_mom_kernel_post_pre_advec_y( double *node_mass_post, const double *post_vol,
                                  const double *density1, double *node_mass_pre, const double *node_flux) {

  node_mass_post[OPS_ACC0(0,0,0)] = 0.125 * ( density1[OPS_ACC2(0,-1,0)] * post_vol[OPS_ACC1(0,-1,0)] +
                                              density1[OPS_ACC2(0,0,0)]   * post_vol[OPS_ACC1(0,0,0)]   +
                                              density1[OPS_ACC2(-1,-1,0)] * post_vol[OPS_ACC1(-1,-1,0)] +
                                              density1[OPS_ACC2(-1,0,0)]  * post_vol[OPS_ACC1(-1,0,0)] +
                                              density1[OPS_ACC2(0,-1,-1)] * post_vol[OPS_ACC1(0,-1,-1)] +
                                              density1[OPS_ACC2(0,0,-1)]   * post_vol[OPS_ACC1(0,0,-1)]   +
                                              density1[OPS_ACC2(-1,-1,-1)] * post_vol[OPS_ACC1(-1,-1,-1)] +
                                              density1[OPS_ACC2(-1,0,-1)]  * post_vol[OPS_ACC1(-1,0,-1)]  );

  node_mass_pre[OPS_ACC3(0,0,0)] = node_mass_post[OPS_ACC0(0,0,0)] - node_flux[OPS_ACC4(0,-1,0)] + node_flux[OPS_ACC4(0,0,0)];
}

inline void advec_mom_kernel_post_pre_advec_z( double *node_mass_post, const double *post_vol,
                                  const double *density1, double *node_mass_pre, const double *node_flux) {

  node_mass_post[OPS_ACC0(0,0,0)] = 0.125 * ( density1[OPS_ACC2(0,-1,0)] * post_vol[OPS_ACC1(0,-1,0)] +
                                              density1[OPS_ACC2(0,0,0)]   * post_vol[OPS_ACC1(0,0,0)]   +
                                              density1[OPS_ACC2(-1,-1,0)] * post_vol[OPS_ACC1(-1,-1,0)] +
                                              density1[OPS_ACC2(-1,0,0)]  * post_vol[OPS_ACC1(-1,0,0)] +
                                              density1[OPS_ACC2(0,-1,-1)] * post_vol[OPS_ACC1(0,-1,-1)] +
                                              density1[OPS_ACC2(0,0,-1)]   * post_vol[OPS_ACC1(0,0,-1)]   +
                                              density1[OPS_ACC2(-1,-1,-1)] * post_vol[OPS_ACC1(-1,-1,-1)] +
                                              density1[OPS_ACC2(-1,0,-1)]  * post_vol[OPS_ACC1(-1,0,-1)]  );

  node_mass_pre[OPS_ACC3(0,0,0)] = node_mass_post[OPS_ACC0(0,0,0)] - node_flux[OPS_ACC4(0,0,-1)] + node_flux[OPS_ACC4(0,0,0)];
}

inline void advec_mom_kernel1_x_nonvector( const double *node_flux, const double *node_mass_pre,
                        double *mom_flux,
                        const double *celldx, const double *vel1) {

  double sigma, wind, width;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  int upwind, donor, downwind, dif;

  double advec_vel_temp;

  if( (node_flux[OPS_ACC0(0,0,0)]) < 0.0) {
    upwind = 2;
    donor = 1;
    downwind = 0;
    dif = donor;
  }
  else {
    upwind = -1;
    donor = 0;
    downwind = 1;
    dif = upwind;
  }

  sigma = fabs(node_flux[OPS_ACC0(0,0,0)])/node_mass_pre[OPS_ACC1(donor,0,0)];

  width = celldx[OPS_ACC3(0,0,0)];
  vdiffuw = vel1[OPS_ACC4(donor,0,0)] - vel1[OPS_ACC4(upwind,0,0)];
  vdiffdw = vel1[OPS_ACC4(downwind,0,0)] - vel1[OPS_ACC4(donor,0,0)];
  limiter=0.0;


  if(vdiffuw*vdiffdw > 0.0) {
    auw = fabs(vdiffuw);
    adw = fabs(vdiffdw);
    wind = 1.0;
    if(vdiffdw <= 0.0) wind = -1.0;
    limiter=wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldx[OPS_ACC3(dif,0,0)])/6.0, MIN(auw, adw));
  }

  advec_vel_temp = vel1[OPS_ACC4(donor,0,0)] + (1.0 - sigma) * limiter;
  mom_flux[OPS_ACC2(0,0,0)] = advec_vel_temp * node_flux[OPS_ACC0(0,0,0)];

}

inline void advec_mom_kernel1_y_nonvector( const double *node_flux, const double *node_mass_pre,
                       double *mom_flux,
                       const double *celldy, const double *vel1) {

  double sigma, wind, width;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  int upwind, donor, downwind, dif;
  double advec_vel_temp;

  if( (node_flux[OPS_ACC0(0,0,0)]) < 0.0) {
    upwind = 2;
    donor = 1;
    downwind = 0;
    dif = donor;
  } else {
    upwind = -1;
    donor = 0;
    downwind = 1;
    dif = upwind;
  }

  sigma = fabs(node_flux[OPS_ACC0(0,0,0)])/node_mass_pre[OPS_ACC1(0,donor,0)];
  width = celldy[OPS_ACC3(0,0,0)];
  vdiffuw = vel1[OPS_ACC4(0,donor,0)] - vel1[OPS_ACC4(0,upwind,0)];
  vdiffdw = vel1[OPS_ACC4(0,downwind,0)] - vel1[OPS_ACC4(0,donor,0)];
  limiter = 0.0;
  if(vdiffuw*vdiffdw > 0.0) {
    auw = fabs(vdiffuw);
    adw = fabs(vdiffdw);
    wind = 1.0;
    if(vdiffdw <= 0.0) wind = -1.0;
    limiter=wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldy[OPS_ACC3(0,dif,0)])/6.0,MIN(auw,adw));
  }
  advec_vel_temp= vel1[OPS_ACC4(0,donor,0)] + (1.0 - sigma) * limiter;
  mom_flux[OPS_ACC2(0,0,0)] = advec_vel_temp * node_flux[OPS_ACC0(0,0,0)];
}

inline void advec_mom_kernel1_z_nonvector( const double *node_flux, const double *node_mass_pre,
                       double *mom_flux,
                       const double *celldz, const double *vel1) {

  double sigma, wind, width;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  int upwind, donor, downwind, dif;
  double advec_vel_temp;

  if( (node_flux[OPS_ACC0(0,0,0)]) < 0.0) {
    upwind = 2;
    donor = 1;
    downwind = 0;
    dif = donor;
  } else {
    upwind = -1;
    donor = 0;
    downwind = 1;
    dif = upwind;
  }

  sigma = fabs(node_flux[OPS_ACC0(0,0,0)])/node_mass_pre[OPS_ACC1(0,0,donor)];
  width = celldz[OPS_ACC3(0,0,0)];
  vdiffuw = vel1[OPS_ACC4(0,0,donor)] - vel1[OPS_ACC4(0,0,upwind)];
  vdiffdw = vel1[OPS_ACC4(0,0,downwind)] - vel1[OPS_ACC4(0,0,donor)];
  limiter = 0.0;
  if(vdiffuw*vdiffdw > 0.0) {
    auw = fabs(vdiffuw);
    adw = fabs(vdiffdw);
    wind = 1.0;
    if(vdiffdw <= 0.0) wind = -1.0;
    limiter=wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldz[OPS_ACC3(0,0,dif)])/6.0,MIN(auw,adw));
  }
  advec_vel_temp= vel1[OPS_ACC4(0,0,donor)] + (1.0 - sigma) * limiter;
  mom_flux[OPS_ACC2(0,0,0)] = advec_vel_temp * node_flux[OPS_ACC0(0,0,0)];
}

inline void advec_mom_kernel2_x(double *vel1, const double *node_mass_post,
                       const  double *node_mass_pre, const double *mom_flux) {

  vel1[OPS_ACC0(0,0,0)] = ( vel1[OPS_ACC0(0,0,0)] * node_mass_pre[OPS_ACC2(0,0,0)]  +
    mom_flux[OPS_ACC3(-1,0,0)] - mom_flux[OPS_ACC3(0,0,0)] ) / node_mass_post[OPS_ACC1(0,0,0)];

}

inline void advec_mom_kernel2_y( double *vel1, const double *node_mass_post,
                        const double *node_mass_pre, const double *mom_flux) {

  vel1[OPS_ACC0(0,0,0)] = ( vel1[OPS_ACC0(0,0,0)] * node_mass_pre[OPS_ACC2(0,0,0)]  +
    mom_flux[OPS_ACC3(0,-1,0)] - mom_flux[OPS_ACC3(0,0,0)] ) / node_mass_post[OPS_ACC1(0,0,0)];
}

inline void advec_mom_kernel2_z( double *vel1, const double *node_mass_post,
                        const double *node_mass_pre, const double *mom_flux) {

  vel1[OPS_ACC0(0,0,0)] = ( vel1[OPS_ACC0(0,0,0)] * node_mass_pre[OPS_ACC2(0,0,0)]  +
    mom_flux[OPS_ACC3(0,0,-1)] - mom_flux[OPS_ACC3(0,0,0)] ) / node_mass_post[OPS_ACC1(0,0,0)];
}
#endif
