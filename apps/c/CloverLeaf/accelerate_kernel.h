#ifndef ACCELERATE_KERNEL_H
#define ACCELERATE_KERNEL_H

#include "data.h"
#include "definitions.h"


void accelerate_kernel( const ACC<double> &density0, const ACC<double> &volume,
                ACC<double> &stepbymass, const ACC<double> &xvel0, ACC<double> &xvel1,
                const ACC<double> &xarea, const ACC<double> &pressure,
                const ACC<double> &yvel0, ACC<double> &yvel1,
                const ACC<double> &yarea, const ACC<double> &viscosity) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0(-1,-1) * volume(-1,-1)
    + density0(0,-1) * volume(0,-1)
    + density0(0,0) * volume(0,0)
    + density0(-1,0) * volume(-1,0) ) * 0.25;

  stepbymass(0,0) = 0.5*dt/ nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel0(0,0) - stepbymass(0,0) *
            ( xarea(0,0)  * ( pressure(0,0) - pressure(-1,0) ) +
              xarea(0,-1) * ( pressure(0,-1) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel0(0,0) - stepbymass(0,0) *
            ( yarea(0,0)  * ( pressure(0,0) - pressure(0,-1) ) +
              yarea(-1,0) * ( pressure(-1,0) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel1(0,0) - stepbymass(0,0) *
            ( xarea(0,0) * ( viscosity(0,0) - viscosity(-1,0) ) +
              xarea(0,-1) * ( viscosity(0,-1) - viscosity(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel1(0,0) - stepbymass(0,0) *
            ( yarea(0,0) * ( viscosity(0,0) - viscosity(0,-1) ) +
              yarea(-1,0) * ( viscosity(-1,0) - viscosity(-1,-1) ) );


}


#endif
