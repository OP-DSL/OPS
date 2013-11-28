#ifndef ADVEC_CELL_KERNEL_H
#define ADVEC_CELL_KERNEL_H
//#include "data.h"
#include "definitions.h"


inline void advec_cell_kernel1_xdir( double *pre_vol, double *post_vol, const double *volume,
                        const double *vol_flux_x, const double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)] +
                           vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,0)]);
  post_vol[OPS_ACC1(0,0)] = pre_vol[OPS_ACC0(0,0)] - ( vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)]);

}



inline void advec_cell_kernel2_xdir( double *pre_vol, double *post_vol, const double *volume,
                        const double *vol_flux_x) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)];
  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)];

}


inline void advec_cell_kernel3_xdir( const double *vol_flux_x, const double *pre_vol, const int *xx,
                              const double *vertexdx,
                              const double *density1, const double *energy1 ,
                              double *mass_flux_x, double *ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, 1,0, -1,0, -2,0};
  //vertexdx accessed with: {0,0, 1,0, -1,0, -2,0};
  //density1, energy1 accessed with: {0,0, 1,0, -1,0, -2,0};
  //xx accessed with: {0,0 ,1,0}

  if(vol_flux_x[OPS_ACC0(0,0)] > 0.0) {
    upwind   = -2; //j-2
    donor    = -1; //j-1
    downwind = 0; //j
    dif      = donor;
  }
  else if (xx[OPS_ACC2(1,0)] < x_max+2) {
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  } else { //*xx[OPS_ACC2(1,0)] >= x_max+2 , then need 0
    upwind   = 0; //xmax+2
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  }
  //return;

  sigmat = fabs(vol_flux_x[OPS_ACC0(0,0)])/pre_vol[OPS_ACC1(donor,0)];
  sigma3 = (1.0 + sigmat)*(vertexdx[OPS_ACC3(0,0)]/vertexdx[OPS_ACC3(dif,0)]);
  sigma4 = 2.0 - sigmat;

  sigma = sigmat;
  sigmav = sigmat;

  diffuw = density1[OPS_ACC4(donor,0)] - density1[OPS_ACC4(upwind,0)];
  diffdw = density1[OPS_ACC4(downwind,0)] - density1[OPS_ACC4(donor,0)];

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_x[OPS_ACC6(0,0)] = (vol_flux_x[OPS_ACC0(0,0)]) * ( density1[OPS_ACC4(donor,0)] + limiter );

  sigmam = fabs(mass_flux_x[OPS_ACC6(0,0)])/( density1[OPS_ACC4(donor,0)] * pre_vol[OPS_ACC1(donor,0)]);
  diffuw = energy1[OPS_ACC5(donor,0)] - energy1[OPS_ACC5(upwind,0)];
  diffdw = energy1[OPS_ACC5(downwind,0)] - energy1[OPS_ACC5(donor,0)];

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux[OPS_ACC7(0,0)] = mass_flux_x[OPS_ACC6(0,0)] * ( energy1[OPS_ACC0(donor,0)] + limiter );
}


inline void advec_cell_kernel4_xdir( double *density1, double *energy1,
                         const double *mass_flux_x, const double *vol_flux_x,
                         double *pre_vol, double *post_vol,
                         double *pre_mass, double *post_mass,
                         double *advec_vol, double *post_ener,
                         double *ener_flux) {

  pre_mass[OPS_ACC6(0,0)] = density1[OPS_ACC0(0,0)] * pre_vol[OPS_ACC4(0,0)];
  post_mass[OPS_ACC7(0,0)] = pre_mass[OPS_ACC6(0,0)] + mass_flux_x[OPS_ACC2(0,0)] - mass_flux_x[OPS_ACC2(1,0)];
  post_ener[OPS_ACC9(0,0)] = ( energy1[OPS_ACC1(0,0)] * pre_mass[OPS_ACC6(0,0)] + ener_flux[OPS_ACC10(0,0)] - ener_flux[OPS_ACC10(1,0)])/post_mass[OPS_ACC7(0,0)];
  advec_vol[OPS_ACC8(0,0)] = pre_vol[OPS_ACC4(0,0)] + vol_flux_x[OPS_ACC3(0,0)] - vol_flux_x[OPS_ACC3(1,0)];
  density1[OPS_ACC0(0,0)] = post_mass[OPS_ACC7(0,0)]/advec_vol[OPS_ACC8(0,0)];
  energy1[OPS_ACC1(0,0)] = post_ener[OPS_ACC9(0,0)];

}


inline void advec_cell_kernel1_ydir( double *pre_vol, double *post_vol, const double *volume,
                        const double *vol_flux_x, const double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + ( vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,0)] +
                           vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)]);
  post_vol[OPS_ACC1(0,0)] = pre_vol[OPS_ACC0(0,0)] - ( vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,0)]);

}

inline void advec_cell_kernel2_ydir( double *pre_vol, double *post_vol, const double *volume,
                        const double *vol_flux_y) {

  pre_vol[OPS_ACC0(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)];
  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)];

}


inline void advec_cell_kernel3_ydir( const double *vol_flux_y, const double *pre_vol, const int *yy,
                              const double *vertexdy,
                              const double *density1, const double *energy1 ,
                              double *mass_flux_y, double *ener_flux) {

  double sigma, sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //vertexdy accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //density1, energy1 accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //yy accessed with: {0,0 ,0,1}

  if(vol_flux_y[OPS_ACC0(0,0)] > 0.0) {
    upwind   = -2; //k-2
    donor    = -1; //k-1
    downwind = 0; //k
    dif      = donor;
  }
  else if (yy[OPS_ACC2(0,1)] < y_max+2) {
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  } else { //*yy[OPS_ACC2(0,1)] >= y_max+2 , then need 0
    upwind   = 0; //ymax+2
    donor    = 0; //k
    downwind = -1; //k-1
    dif      = upwind;
  }
  //return;

  sigmat = fabs(vol_flux_y[OPS_ACC0(0,0)])/pre_vol[OPS_ACC1(0,donor)];
  sigma3 = (1.0 + sigmat)*(vertexdy[OPS_ACC3(0,0)]/vertexdy[OPS_ACC3(0,dif)]);
  sigma4 = 2.0 - sigmat;

  sigma = sigmat;
  sigmav = sigmat;

  diffuw = density1[OPS_ACC4(0,donor)] - density1[OPS_ACC4(0,upwind)];
  diffdw = density1[OPS_ACC4(0,downwind)] - density1[OPS_ACC4(0,donor)];

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_y[OPS_ACC6(0,0)] = (vol_flux_y[OPS_ACC0(0,0)]) * ( density1[OPS_ACC4(0,donor)] + limiter );

  sigmam = fabs(mass_flux_y[OPS_ACC6(0,0)])/( density1[OPS_ACC4(0,donor)] * pre_vol[OPS_ACC1(0,donor)]);
  diffuw = energy1[OPS_ACC5(0,donor)] - energy1[OPS_ACC5(0,upwind)];
  diffdw = energy1[OPS_ACC5(0,downwind)] - energy1[OPS_ACC5(0,donor)];

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux[OPS_ACC7(0,0)] = mass_flux_y[OPS_ACC6(0,0)] * ( energy1[OPS_ACC0(0,donor)] + limiter );
}

inline void advec_cell_kernel4_ydir( double *density1, double *energy1,
                         const double *mass_flux_y, const double *vol_flux_y,
                         double *pre_vol, double *post_vol,
                         double *pre_mass, double *post_mass,
                         double *advec_vol, double *post_ener,
                         double *ener_flux) {

  pre_mass[OPS_ACC6(0,0)] = density1[OPS_ACC0(0,0)] * pre_vol[OPS_ACC4(0,0)];
  post_mass[OPS_ACC7(0,0)] = pre_mass[OPS_ACC6(0,0)] + mass_flux_y[OPS_ACC2(0,0)] - mass_flux_y[OPS_ACC2(0,1)];
  post_ener[OPS_ACC9(0,0)] = ( energy1[OPS_ACC1(0,0)] * pre_mass[OPS_ACC6(0,0)] + ener_flux[OPS_ACC10(0,0)] - ener_flux[OPS_ACC10(0,1)])/post_mass[OPS_ACC7(0,0)];
  advec_vol[OPS_ACC8(0,0)] = pre_vol[OPS_ACC4(0,0)] + vol_flux_y[OPS_ACC3(0,0)] - vol_flux_y[OPS_ACC3(0,1)];
  density1[OPS_ACC0(0,0)] = post_mass[OPS_ACC7(0,0)]/advec_vol[OPS_ACC8(0,0)];
  energy1[OPS_ACC1(0,0)] = post_ener[OPS_ACC9(0,0)];

}


#endif
