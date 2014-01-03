#ifndef ACCELERATE_KERNEL_H
#define ACCELERATE_KERNEL_H

////#include "data.h"
#include "definitions.h"




inline void accelerate_kernel_stepbymass(const double *density0, const double *volume,
                double *stepbymass) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0[OPS_ACC0(-1,-1)] * volume[OPS_ACC1(-1,-1)]
    + density0[OPS_ACC0(0,-1)] * volume[OPS_ACC1(0,-1)]
    + density0[OPS_ACC0(0,0)] * volume[OPS_ACC1(0,0)]
    + density0[OPS_ACC0(-1,0)] * volume[OPS_ACC1(-1,0)] ) * 0.25;

  //stepbymass[OPS_ACC2(0,0)] = 0.5*dt / nodal_mass;
  stepbymass[OPS_ACC2(0,0)] = 0.5*dt / nodal_mass;

}


inline void accelerate_kernelx1(const  double *xvel0, double *xvel1,
                        const double *stepbymass,
                        const double *xarea, const double *pressure) {
  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1[OPS_ACC1(0,0)] = xvel0[OPS_ACC0(0,0)] - stepbymass[OPS_ACC2(0,0)] *
            ( xarea[OPS_ACC3(0,0)]  * ( pressure[OPS_ACC4(0,0)] - pressure[OPS_ACC4(-1,0)] ) +
              xarea[OPS_ACC3(0,-1)] * ( pressure[OPS_ACC4(0,-1)] - pressure[OPS_ACC4(-1,-1)] ) );
}


inline void accelerate_kernely1( const double *yvel0, double *yvel1,
                        const double *stepbymass,
                        const double *yarea, const double *pressure) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1[OPS_ACC1(0,0)] = yvel0[OPS_ACC0(0,0)] - stepbymass[OPS_ACC2(0,0)] *
            ( yarea[OPS_ACC3(0,0)]  * ( pressure[OPS_ACC4(0,0)] - pressure[OPS_ACC4(0,-1)] ) +
              yarea[OPS_ACC3(-1,0)] * ( pressure[OPS_ACC4(-1,0)] - pressure[OPS_ACC4(-1,-1)] ) );

}



inline void accelerate_kernelx2( double *xvel1, const double *stepbymass,
                        const double *xarea, const double *viscosity) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1[OPS_ACC0(0,0)] = xvel1[OPS_ACC0(0,0)] - stepbymass[OPS_ACC1(0,0)] *
            ( xarea[OPS_ACC2(0,0)] * ( viscosity[OPS_ACC3(0,0)] - viscosity[OPS_ACC3(-1,0)] ) +
              xarea[OPS_ACC2(0,-1)] * ( viscosity[OPS_ACC3(0,-1)] - viscosity[OPS_ACC3(-1,-1)] ) );
}



inline void accelerate_kernely2( double *yvel1, const double *stepbymass,
                        const double *yarea, const double *viscosity) {

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1[OPS_ACC0(0,0)] = yvel1[OPS_ACC0(0,0)] - stepbymass[OPS_ACC1(0,0)] *
            ( yarea[OPS_ACC2(0,0)] * ( viscosity[OPS_ACC3(0,0)] - viscosity[OPS_ACC3(0,-1)] ) +
              yarea[OPS_ACC2(-1,0)] * ( viscosity[OPS_ACC3(-1,0)] - viscosity[OPS_ACC3(-1,-1)] ) );

}
#endif
