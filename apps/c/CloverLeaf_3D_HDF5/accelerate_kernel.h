#ifndef ACCELERATE_KERNEL_H
#define ACCELERATE_KERNEL_H

#include "data.h"
#include "definitions.h"


void accelerate_kernel( const double *density0, const double *volume,
                double *stepbymass, const double *xvel0, double *xvel1,
                const double *xarea, const double *pressure,
                const double *yvel0, double *yvel1,
                const double *yarea, const double *viscosity,
                const double *zvel0, double *zvel1,
                const double *zarea) {

  double nodal_mass = 0.0;
  nodal_mass =(density0[OPS_ACC0(-1,-1, 0)] * volume[OPS_ACC1(-1,-1, 0)] +
               density0[OPS_ACC0( 0,-1, 0)] * volume[OPS_ACC1( 0,-1, 0)] +
               density0[OPS_ACC0( 0, 0, 0)] * volume[OPS_ACC1( 0, 0, 0)] +
               density0[OPS_ACC0(-1, 0, 0)] * volume[OPS_ACC1(-1, 0, 0)] +
               density0[OPS_ACC0(-1,-1,-1)] * volume[OPS_ACC1(-1,-1,-1)] +
               density0[OPS_ACC0( 0,-1,-1)] * volume[OPS_ACC1( 0,-1,-1)] +
               density0[OPS_ACC0( 0, 0,-1)] * volume[OPS_ACC1( 0, 0,-1)] +
               density0[OPS_ACC0(-1, 0,-1)] * volume[OPS_ACC1(-1, 0,-1)]) * 0.125;

  stepbymass[OPS_ACC2(0,0,0)] = 0.25*dt / nodal_mass;

  xvel1[OPS_ACC4(0,0,0)] = xvel0[OPS_ACC3(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( xarea[OPS_ACC5(0,0,0)]  * ( pressure[OPS_ACC6(0,0,0)] - pressure[OPS_ACC6(-1,0,0)] ) +
              xarea[OPS_ACC5(0,-1,0)] * ( pressure[OPS_ACC6(0,-1,0)] - pressure[OPS_ACC6(-1,-1,0)] ) +
              xarea[OPS_ACC5(0,0,-1)] * ( pressure[OPS_ACC6(0,0,-1)] - pressure[OPS_ACC6(-1,0,-1)] ) +
              xarea[OPS_ACC5(0,-1,-1)] * ( pressure[OPS_ACC6(0,-1,-1)] - pressure[OPS_ACC6(-1,-1,-1)] ) );

  yvel1[OPS_ACC8(0,0,0)] = yvel0[OPS_ACC7(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( yarea[OPS_ACC9(0,0,0)]  * ( pressure[OPS_ACC6(0,0,0)] - pressure[OPS_ACC6(0,-1,0)] ) +
              yarea[OPS_ACC9(-1,0,0)] * ( pressure[OPS_ACC6(-1,0,0)] - pressure[OPS_ACC6(-1,-1,0)] ) +
              yarea[OPS_ACC9(0,0,-1)] * ( pressure[OPS_ACC6(0,0,-1)] - pressure[OPS_ACC6(0,-1,-1)] ) +
              yarea[OPS_ACC9(-1,0,-1)]* ( pressure[OPS_ACC6(-1,0,-1)] - pressure[OPS_ACC6(-1,-1,-1)] ) );

  zvel1[OPS_ACC12(0,0,0)] = zvel0[OPS_ACC11(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( zarea[OPS_ACC13(0,0,0)]  * ( pressure[OPS_ACC6(0,0,0)] - pressure[OPS_ACC6(0,0,-1)] ) +
              zarea[OPS_ACC13(0,-1,0)] * ( pressure[OPS_ACC6(0,-1,0)] - pressure[OPS_ACC6(0,-1,-1)] ) +
              zarea[OPS_ACC13(-1,0,0)] * ( pressure[OPS_ACC6(-1,0,0)] - pressure[OPS_ACC6(-1,0,-1)] ) +
              zarea[OPS_ACC13(-1,-1,0)]* ( pressure[OPS_ACC6(-1,-1,0)] - pressure[OPS_ACC6(-1,-1,-1)] ) );


  xvel1[OPS_ACC4(0,0,0)] = xvel1[OPS_ACC4(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( xarea[OPS_ACC5(0,0,0)]  * ( viscosity[OPS_ACC10(0,0,0)] - viscosity[OPS_ACC10(-1,0,0)] ) +
              xarea[OPS_ACC5(0,-1,0)] * ( viscosity[OPS_ACC10(0,-1,0)] - viscosity[OPS_ACC10(-1,-1,0)] ) +
              xarea[OPS_ACC5(0,0,-1)] * ( viscosity[OPS_ACC10(0,0,-1)] - viscosity[OPS_ACC10(-1,0,-1)] ) +
              xarea[OPS_ACC5(0,-1,-1)]* ( viscosity[OPS_ACC10(0,-1,-1)] - viscosity[OPS_ACC10(-1,-1,-1)] ) );

  yvel1[OPS_ACC8(0,0,0)] = yvel1[OPS_ACC8(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( yarea[OPS_ACC9(0,0,0)]  * ( viscosity[OPS_ACC10(0,0,0)]  - viscosity[OPS_ACC10(0,-1,0)] ) +
              yarea[OPS_ACC9(-1,0,0)] * ( viscosity[OPS_ACC10(-1,0,0)] - viscosity[OPS_ACC10(-1,-1,0)] ) +
              yarea[OPS_ACC9(0,0,-1)] * ( viscosity[OPS_ACC10(0,0,-1)] - viscosity[OPS_ACC10(0,-1,-1)] ) +
              yarea[OPS_ACC9(-1,0,-1)]* ( viscosity[OPS_ACC10(-1,0,-1)]- viscosity[OPS_ACC10(-1,-1,-1)] ) );

  zvel1[OPS_ACC12(0,0,0)] = zvel1[OPS_ACC12(0,0,0)] - stepbymass[OPS_ACC2(0,0,0)] *
            ( zarea[OPS_ACC13(0,0,0)]  * ( viscosity[OPS_ACC10(0,0,0)]  - viscosity[OPS_ACC10(0,0,-1)] ) +
              zarea[OPS_ACC13(0,-1,0)] * ( viscosity[OPS_ACC10(0,-1,0)] - viscosity[OPS_ACC10(0,-1,-1)] ) +
              zarea[OPS_ACC13(-1,0,0)] * ( viscosity[OPS_ACC10(-1,0,0)] - viscosity[OPS_ACC10(-1,0,-1)] ) +
              zarea[OPS_ACC13(-1,-1,0)]* ( viscosity[OPS_ACC10(-1,-1,0)]- viscosity[OPS_ACC10(-1,-1,-1)] ) );


}

#endif