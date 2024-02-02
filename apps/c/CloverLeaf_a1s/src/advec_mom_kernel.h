#ifndef ADVEC_MOM_KERNEL_H
#define ADVEC_MOM_KERNEL_H

inline void advec_mom_kernel_x1( ACC<double> &pre_vol, ACC<double> &post_vol,
                          const ACC<double> &volume,
                          const ACC<double> &vol_flux_x, const ACC<double> &vol_flux_y) {

  post_vol(0,0) = volume(0,0) + vol_flux_y(0,1) -  vol_flux_y(0,0);
  pre_vol(0,0) = post_vol(0,0) + vol_flux_x(1,0) - vol_flux_x(0,0);

}

inline void advec_mom_kernel_y1( ACC<double> &pre_vol, ACC<double> &post_vol,
                          const ACC<double> &volume,
                          const ACC<double> &vol_flux_x, const ACC<double> &vol_flux_y) {

  post_vol(0,0) = volume(0,0) + vol_flux_x(1,0) -  vol_flux_x(0,0);
  pre_vol(0,0) = post_vol(0,0) + vol_flux_y(0,1) - vol_flux_y(0,0);

}

inline void advec_mom_kernel_x2( ACC<double> &pre_vol, ACC<double> &post_vol,
                          const ACC<double> &volume,
                          const ACC<double> &vol_flux_y) {

  post_vol(0,0)  = volume(0,0) ;
  pre_vol(0,0)   = post_vol(0,0)  + vol_flux_y(0,1) - vol_flux_y(0,0);

}

inline void advec_mom_kernel_y2( ACC<double> &pre_vol, ACC<double> &post_vol,
                          const ACC<double> &volume,
                          const ACC<double> &vol_flux_x) {

  post_vol(0,0)  = volume(0,0) ;
  pre_vol(0,0)   = post_vol(0,0)  + vol_flux_x(1,0) - vol_flux_x(0,0);

}

inline void advec_mom_kernel_mass_flux_x( ACC<double> &node_flux, const ACC<double> &mass_flux_x) {

  //mass_flux_x accessed with: {0,0, 1,0, 0,-1, 1,-1}

  node_flux(0,0) = 0.25 * ( mass_flux_x(0,-1) + mass_flux_x(0,0) +
    mass_flux_x(1,-1) + mass_flux_x(1,0) ); // Mass Flux in x
}


inline void advec_mom_kernel_mass_flux_y( ACC<double> &node_flux, const ACC<double> &mass_flux_y) {

  //mass_flux_y accessed with: {0,0, 0,1, -1,0, -1,1}

  node_flux(0,0) = 0.25 * ( mass_flux_y(-1,0) + mass_flux_y(0,0) +
      mass_flux_y(-1,1) + mass_flux_y(0,1) ); // Mass Flux in y
}


inline void advec_mom_kernel_post_pre_advec_x( ACC<double> &node_mass_post, const ACC<double> &post_vol,
                                  const ACC<double> &density1, ACC<double> &node_mass_pre, const ACC<double> &node_flux) {

  //post_vol accessed with: {0,0, -1,0, 0,-1, -1,-1}
  //density1 accessed with: {0,0, -1,0, 0,-1, -1,-1}

  node_mass_post(0,0) = 0.25 * ( density1(0,-1) * post_vol(0,-1) +
                              density1(0,0)   * post_vol(0,0)   +
                              density1(-1,-1) * post_vol(-1,-1) +
                              density1(-1,0)  * post_vol(-1,0)  );

  //node_flux accessed with: {0,0, -1,0}
  node_mass_pre(0,0) = node_mass_post(0,0) - node_flux(-1,0) + node_flux(0,0);

}

//this is the same as advec_mom_kernel_post_advec_x ... just repeated here for debugging
inline void advec_mom_kernel_post_pre_advec_y( ACC<double> &node_mass_post, const ACC<double> &post_vol,
                                  const ACC<double> &density1, ACC<double> &node_mass_pre, const ACC<double> &node_flux) {

  //post_vol accessed with: {0,0, -1,0, 0,-1, -1,-1}
  //density1 accessed with: {0,0, -1,0, 0,-1, -1,-1}

  node_mass_post(0,0) = 0.25 * ( density1(0,-1) * post_vol(0,-1) +
                              density1(0,0)   * post_vol(0,0)   +
                              density1(-1,-1) * post_vol(-1,-1) +
                              density1(-1,0)  * post_vol(-1,0)  );

  //node_flux accessed with: {0,0, 0,-1}
  node_mass_pre(0,0) = node_mass_post(0,0) - node_flux(0,-1) + node_flux(0,0);


}

inline void advec_mom_kernel1_x( const ACC<double> &node_flux, const ACC<double> &node_mass_pre,
                        ACC<double> &advec_vel, ACC<double> &mom_flux,
                        const ACC<double> &celldx, const ACC<double> &vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 1,0}
  //celldx is accessed with {0,0, 1,0, -1,0, -2,0} striding in x
  //vel1 is accessed with {0,0, 1,0, 2,0, -1,0}

  double sigma, wind, width;
  double sigma2, wind2;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  double vdiffuw2, vdiffdw2, auw2, limiter2;

  sigma = fabs(node_flux(0,0))/node_mass_pre(1,0);
  sigma2 = fabs(node_flux(0,0))/node_mass_pre(0,0);

  width = celldx(0,0);
  vdiffuw = vel1(1,0) - vel1(2,0);
  vdiffdw = vel1(0,0) - vel1(1,0);
  vdiffuw2 = vel1(0,0) - vel1(-1,0);
  vdiffdw2 = -vdiffdw;

  auw = fabs(vdiffuw);
  adw = fabs(vdiffdw);
  auw2 = fabs(vdiffuw2);
  wind = 1.0;
  wind2 = 1.0;

  if(vdiffdw <= 0.0) wind = -1.0;
  if(vdiffdw2 <= 0.0) wind2 = -1.0;

  limiter = wind * MIN( width * ( (2.0 - sigma) * adw/width + (1.0 + sigma) *
                        auw/celldx(1,0) )/6.0 , MIN(auw, adw) );
  limiter2= wind2* MIN( width * ( (2.0 - sigma2) * adw/width + (1.0 + sigma2) *
                        auw2/celldx(-1,0) )/6.0, MIN(auw2,adw) );

  if((vdiffuw * vdiffdw) <= 0.0) limiter = 0.0;
  if((vdiffuw2 * vdiffdw2) <= 0.0) limiter2 = 0.0;

  if( (node_flux(0,0)) < 0.0) {
    advec_vel(0,0) = vel1(1,0) + (1.0 - sigma) * limiter;
  }
  else {
    advec_vel(0,0) = vel1(0,0) + (1.0 - sigma2) * limiter2;
  }

  mom_flux(0,0) = advec_vel(0,0) * node_flux(0,0);

}


inline void advec_mom_kernel1_x_nonvector( const ACC<double> &node_flux, const ACC<double> &node_mass_pre,
                        ACC<double> &mom_flux,
                        const ACC<double> &celldx, const ACC<double> &vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 1,0}
  //celldx is accessed with {0,0, 1,0, -1,0} striding in x
  //vel1 is accessed with {0,0, 1,0, 2,0, -1,0}

  double sigma, wind, width;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  int upwind, donor, downwind, dif;

  double advec_vel_temp;

  if( (node_flux(0,0)) < 0.0) {
    upwind = 2;
    donor =1;
    downwind = 0;
    dif = donor;
  }
  else {
    upwind=-1;
    donor=0;
    downwind=1;
    dif=upwind;
  }

  sigma = fabs(node_flux(0,0))/node_mass_pre(donor,0);

  width = celldx(0,0);
  vdiffuw = vel1(donor,0) - vel1(upwind,0);
  vdiffdw = vel1(downwind,0) - vel1(donor,0);
  limiter=0.0;


  if(vdiffuw*vdiffdw > 0.0) {
    auw = fabs(vdiffuw);
    adw = fabs(vdiffdw);
    wind = 1.0;
    if(vdiffdw <= 0.0) wind = -1.0;
    limiter=wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldx(dif,0))/6.0, MIN(auw, adw));
  }

  advec_vel_temp = vel1(donor,0) + (1.0 - sigma) * limiter;
  mom_flux(0,0) = advec_vel_temp * node_flux(0,0);

}



inline void advec_mom_kernel1_y( const ACC<double> &node_flux, const ACC<double> &node_mass_pre,
                        ACC<double> &advec_vel, ACC<double> &mom_flux,
                        const ACC<double> &celldy, const ACC<double> &vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 0,1}
  //celldy is accessed with {0,0, 0,1, 0,-1, 0,-2} striding in y
  //vel1 is accessed with {0,0, 0,1, 0,2, 0,-1}

  double sigma, wind, width;
  double sigma2, wind2;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  double vdiffuw2, vdiffdw2, auw2, limiter2;

  sigma = fabs(node_flux(0,0))/node_mass_pre(0,1);
  sigma2 = fabs(node_flux(0,0))/node_mass_pre(0,0);

  width = celldy(0,0);
  vdiffuw = vel1(0,1) - vel1(0,2);
  vdiffdw = vel1(0,0) - vel1(0,1);
  vdiffuw2 = vel1(0,0) - vel1(0,-1);
  vdiffdw2 = -vdiffdw;

  auw = fabs(vdiffuw);
  adw = fabs(vdiffdw);
  auw2 = fabs(vdiffuw2);
  wind = 1.0;
  wind2 = 1.0;

  if(vdiffdw <= 0.0) wind = -1.0;
  if(vdiffdw2 <= 0.0) wind2 = -1.0;

  limiter = wind * MIN( width * ( (2.0 - sigma) * adw/width + (1.0 + sigma) *
                        auw/celldy(0,1) )/6.0 , MIN(auw, adw) );
  limiter2= wind2* MIN( width * ( (2.0 - sigma2) * adw/width + (1.0 + sigma2) *
                        auw2/celldy(0,-1) )/6.0, MIN(auw2,adw) );

  if((vdiffuw * vdiffdw) <= 0.0) limiter = 0.0;
  if((vdiffuw2 * vdiffdw2) <= 0.0) limiter2 = 0.0;

  if( (node_flux(0,0)) < 0.0) {
    advec_vel(0,0) = vel1(0,1) + (1.0 - sigma) * limiter;
  }
  else {
    advec_vel(0,0) = vel1(0,0) + (1.0 - sigma2) * limiter2;
  }

  mom_flux(0,0) = advec_vel(0,0) * node_flux(0,0);
}


inline void advec_mom_kernel1_y_nonvector( const ACC<double> &node_flux, const ACC<double> &node_mass_pre,
                       ACC<double> &mom_flux,
                       const ACC<double> &celldy, const ACC<double> &vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 0,1}
  //celldy is accessed with {0,0, 0,1, 0,-1} striding in y
  //vel1 is accessed with {0,0, 0,1, 0,2, 0,-1}

  double sigma, wind, width;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  int upwind, donor, downwind, dif;
  double advec_vel_temp;

  if( (node_flux(0,0)) < 0.0) {
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

  sigma = fabs(node_flux(0,0))/node_mass_pre(0,donor);
  width = celldy(0,0);
  vdiffuw = vel1(0,donor) - vel1(0,upwind);
  vdiffdw = vel1(0,downwind) - vel1(0,donor);
  limiter = 0.0;
  if(vdiffuw*vdiffdw > 0.0) {
    auw = fabs(vdiffuw);
    adw = fabs(vdiffdw);
    wind = 1.0;
    if(vdiffdw <= 0.0) wind = -1.0;
    limiter=wind*MIN(width*((2.0-sigma)*adw/width+(1.0+sigma)*auw/celldy(0,dif))/6.0,MIN(auw,adw));
  }
  advec_vel_temp= vel1(0,donor) + (1.0 - sigma) * limiter;
  mom_flux(0,0) = advec_vel_temp * node_flux(0,0);
}



inline void advec_mom_kernel2_x(ACC<double> &vel1, const ACC<double> &node_mass_post,
                       const  ACC<double> &node_mass_pre, const ACC<double> &mom_flux) {

  //mom_flux accessed with: {0,0, -1,0}
  vel1(0,0) = ( vel1(0,0) * node_mass_pre(0,0)  +
    mom_flux(-1,0) - mom_flux(0,0) ) / node_mass_post(0,0);

}

inline void advec_mom_kernel2_y( ACC<double> &vel1, const ACC<double> &node_mass_post,
                        const ACC<double> &node_mass_pre, const ACC<double> &mom_flux) {

  //mom_flux accessed with: {0,0, 0,-1}
  vel1(0,0) = ( vel1(0,0) * node_mass_pre(0,0)  +
    mom_flux(0,-1) - mom_flux(0,0) ) / node_mass_post(0,0);
}




#endif
