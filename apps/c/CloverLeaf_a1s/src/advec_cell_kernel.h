#ifndef ADVEC_CELL_KERNEL_H
#define ADVEC_CELL_KERNEL_H
//#include "data.h"
#include "definitions.h"


inline void advec_cell_kernel1_xdir( ACC<double> &pre_vol, ACC<double> &post_vol, const ACC<double> &volume,
                        const ACC<double> &vol_flux_x, const ACC<double> &vol_flux_y) {

  pre_vol(0,0) = volume(0,0) + ( vol_flux_x(1,0) - vol_flux_x(0,0) +
                           vol_flux_y(0,1) - vol_flux_y(0,0));
  post_vol(0,0) = pre_vol(0,0) - ( vol_flux_x(1,0) - vol_flux_x(0,0));

}



inline void advec_cell_kernel2_xdir( ACC<double> &pre_vol, ACC<double> &post_vol, const ACC<double> &volume,
                        const ACC<double> &vol_flux_x) {

  pre_vol(0,0) = volume(0,0) + vol_flux_x(1,0) - vol_flux_x(0,0);
  post_vol(0,0) = volume(0,0);

}


inline void advec_cell_kernel3_xdir( const ACC<double> &vol_flux_x, const ACC<double> &pre_vol, const ACC<int> &xx,
                              const ACC<double> &vertexdx,
                              const ACC<double> &density1, const ACC<double> &energy1 ,
                              ACC<double> &mass_flux_x, ACC<double> &ener_flux) {

  double sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int x_max=field.x_max;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, -1,0};
  //vertexdx accessed with: {0,0, 1,0, -1,0};
  //density1, energy1 accessed with: {0,0, 1,0, -1,0, -2,0};
  //xx accessed with: {0,0 ,1,0}

  if(vol_flux_x(0,0) > 0.0) {
    upwind   = -2; //j-2
    donor    = -1; //j-1
    downwind = 0; //j
    dif      = donor;
  }
  else if (xx(1,0) < x_max+2-2) { //extra -2 due to extraborder in OPS
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  } else { //*xx(1,0) >= x_max+2 , then need 0
    upwind   = 0; //xmax+2
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  }
  //return;

  sigmat = fabs(vol_flux_x(0,0))/pre_vol(donor,0);
  sigma3 = (1.0 + sigmat)*(vertexdx(0,0)/vertexdx(dif,0));
  sigma4 = 2.0 - sigmat;

  sigmav = sigmat;

  diffuw = density1(donor,0) - density1(upwind,0);
  diffdw = density1(downwind,0) - density1(donor,0);

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_x(0,0) = (vol_flux_x(0,0)) * ( density1(donor,0) + limiter );

  sigmam = fabs(mass_flux_x(0,0))/( density1(donor,0) * pre_vol(donor,0));
  diffuw = energy1(donor,0) - energy1(upwind,0);
  diffdw = energy1(downwind,0) - energy1(donor,0);

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux(0,0) = mass_flux_x(0,0) * ( energy1(donor,0) + limiter );
}


inline void advec_cell_kernel4_xdir( ACC<double> &density1, ACC<double> &energy1,
                         const ACC<double> &mass_flux_x, const ACC<double> &vol_flux_x,
                         const ACC<double> &pre_vol, const ACC<double> &post_vol,
                         ACC<double> &pre_mass, ACC<double> &post_mass,
                         ACC<double> &advec_vol, ACC<double> &post_ener,
                         const ACC<double> &ener_flux) {

  pre_mass(0,0) = density1(0,0) * pre_vol(0,0);
  post_mass(0,0) = pre_mass(0,0) + mass_flux_x(0,0) - mass_flux_x(1,0);
  post_ener(0,0) = ( energy1(0,0) * pre_mass(0,0) + ener_flux(0,0) - ener_flux(1,0))/post_mass(0,0);
  advec_vol(0,0) = pre_vol(0,0) + vol_flux_x(0,0) - vol_flux_x(1,0);
  density1(0,0) = post_mass(0,0)/advec_vol(0,0);
  energy1(0,0) = post_ener(0,0);

}

inline void advec_cell_kernel1_ydir( ACC<double> &pre_vol, ACC<double> &post_vol, const ACC<double> &volume,
                        const ACC<double> &vol_flux_x, const ACC<double> &vol_flux_y) {

  pre_vol(0,0) = volume(0,0) + ( vol_flux_y(0,1) - vol_flux_y(0,0) +
                           vol_flux_x(1,0) - vol_flux_x(0,0));
  post_vol(0,0) = pre_vol(0,0) - ( vol_flux_y(0,1) - vol_flux_y(0,0));

}

inline void advec_cell_kernel2_ydir( ACC<double> &pre_vol, ACC<double> &post_vol, const ACC<double> &volume,
                        const ACC<double> &vol_flux_y) {

  pre_vol(0,0) = volume(0,0) + vol_flux_y(0,1) - vol_flux_y(0,0);
  post_vol(0,0) = volume(0,0);

}


inline void advec_cell_kernel3_ydir( const ACC<double> &vol_flux_y, const ACC<double> &pre_vol, const ACC<int> &yy,
                              const ACC<double> &vertexdy,
                              const ACC<double> &density1, const ACC<double> &energy1 ,
                              ACC<double> &mass_flux_y, ACC<double> &ener_flux) {

  double sigmat, sigmav, sigmam, sigma3, sigma4;
  double diffuw, diffdw, limiter;
  double one_by_six = 1.0/6.0;

  int y_max=field.y_max;

  int upwind,donor,downwind,dif;

  //pre_vol accessed with: {0,0, 0,1};
  //vertexdy accessed with: {0,0, 0,1, 0,-1};
  //density1, energy1 accessed with: {0,0, 0,1, 0,-1, 0,-2};
  //yy accessed with: {0,0 ,0,1}

  if(vol_flux_y(0,0) > 0.0) {
    upwind   = -2; //k-2
    donor    = -1; //k-1
    downwind = 0; //k
    dif      = donor;
  }
  else if (yy(0,1) < y_max+2-2) { //extra -2 due to extra border in OPS version
    upwind   = 1; //j+1
    donor    = 0; //j
    downwind = -1; //j-1
    dif      = upwind;
  } else { //*yy(0,1) >= y_max+2 , then need 0
    upwind   = 0; //ymax+2
    donor    = 0; //k
    downwind = -1; //k-1
    dif      = upwind;
  }
  //return;

  sigmat = fabs(vol_flux_y(0,0))/pre_vol(0,donor);
  sigma3 = (1.0 + sigmat)*(vertexdy(0,0)/vertexdy(0,dif));
  sigma4 = 2.0 - sigmat;

  sigmav = sigmat;

  diffuw = density1(0,donor) - density1(0,upwind);
  diffdw = density1(0,downwind) - density1(0,donor);

  if( (diffuw*diffdw) > 0.0)
    limiter=(1.0 - sigmav) * SIGN(1.0 , diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3*fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  mass_flux_y(0,0) = (vol_flux_y(0,0)) * ( density1(0,donor) + limiter );

  sigmam = fabs(mass_flux_y(0,0))/( density1(0,donor) * pre_vol(0,donor));
  diffuw = energy1(0,donor) - energy1(0,upwind);
  diffdw = energy1(0,downwind) - energy1(0,donor);

  if( (diffuw*diffdw) > 0.0)
    limiter = (1.0 - sigmam) * SIGN(1.0,diffdw) *
    MIN( MIN(fabs(diffuw), fabs(diffdw)),
    one_by_six * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw)));
  else
    limiter=0.0;

  ener_flux(0,0) = mass_flux_y(0,0) * ( energy1(0,donor) + limiter );
}

inline void advec_cell_kernel4_ydir( ACC<double> &density1, ACC<double> &energy1,
                         const ACC<double> &mass_flux_y, const ACC<double> &vol_flux_y,
                         const ACC<double> &pre_vol, const ACC<double> &post_vol,
                         ACC<double> &pre_mass, ACC<double> &post_mass,
                         ACC<double> &advec_vol, ACC<double> &post_ener,
                         const ACC<double> &ener_flux) {

  pre_mass(0,0) = density1(0,0) * pre_vol(0,0);
  post_mass(0,0) = pre_mass(0,0) + mass_flux_y(0,0) - mass_flux_y(0,1);
  post_ener(0,0) = ( energy1(0,0) * pre_mass(0,0) + ener_flux(0,0) - ener_flux(0,1))/post_mass(0,0);
  advec_vol(0,0) = pre_vol(0,0) + vol_flux_y(0,0) - vol_flux_y(0,1);
  density1(0,0) = post_mass(0,0)/advec_vol(0,0);
  energy1(0,0) = post_ener(0,0);

}


#endif
