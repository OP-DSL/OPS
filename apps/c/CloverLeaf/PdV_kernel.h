#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

#include "data.h"
#include "definitions.h"

void PdV_kernel_predict(const ACC<double> &xarea, const ACC<double> &xvel0,
                const ACC<double> &yarea, const ACC<double> &yvel0,
                ACC<double> &volume_change, const ACC<double> &volume,
                const ACC<double> &pressure,
                const ACC<double> &density0, ACC<double> &density1,
                const ACC<double> &viscosity,
                const ACC<double> &energy0, ACC<double> &energy1) {

  //xvel0, S2D_00_P10_0P1_P1P1

  double recip_volume, energy_change;//, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea(0,0) * ( xvel0(0,0) + xvel0(0,1) +
                                xvel0(0,0) + xvel0(0,1) ) ) * 0.25 * dt * 0.5;
  right_flux = ( xarea(1,0) * ( xvel0(1,0) + xvel0(1,1) +
                                 xvel0(1,0) + xvel0(1,1) ) ) * 0.25 * dt * 0.5;

  bottom_flux = ( yarea(0,0) * ( yvel0(0,0) + yvel0(1,0) +
                                  yvel0(0,0) + yvel0(1,0) ) ) * 0.25* dt * 0.5;
  top_flux = ( yarea(0,1) * ( yvel0(0,1) + yvel0(1,1) +
                               yvel0(0,1) + yvel0(1,1) ) ) * 0.25 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change(0,0) = (volume(0,0))/(volume(0,0) + total_flux);

  //min_cell_volume = MIN( volume(0,0) + right_flux - left_flux + top_flux - bottom_flux ,
  //                         MIN(volume(0,0) + right_flux - left_flux,
  //                         volume(0,0) + top_flux - bottom_flux) );

  recip_volume = 1.0/volume(0,0);

  energy_change = ( pressure(0,0)/density0(0,0) +
                    viscosity(0,0)/density0(0,0) ) * total_flux * recip_volume;
  energy1(0,0) = energy0(0,0) - energy_change;
  density1(0,0) = density0(0,0) * volume_change(0,0);

}

void PdV_kernel_nopredict(const ACC<double> &xarea, const ACC<double> &xvel0, const ACC<double> &xvel1,
                const ACC<double> &yarea, const ACC<double> &yvel0, const ACC<double> &yvel1,
                ACC<double> &volume_change, const ACC<double> &volume,
                const ACC<double> &pressure,
                const ACC<double> &density0, ACC<double> &density1,
                const ACC<double> &viscosity,
                const ACC<double> &energy0, ACC<double> &energy1) {

  //xvel0, S2D_00_P10_0P1_P1P1

  double recip_volume, energy_change;//, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea(0,0) * ( xvel0(0,0) + xvel0(0,1) +
                                xvel1(0,0) + xvel1(0,1) ) ) * 0.25 * dt;
  right_flux = ( xarea(1,0) * ( xvel0(1,0) + xvel0(1,1) +
                                 xvel1(1,0) + xvel1(1,1) ) ) * 0.25 * dt;

  bottom_flux = ( yarea(0,0) * ( yvel0(0,0) + yvel0(1,0) +
                                  yvel1(0,0) + yvel1(1,0) ) ) * 0.25* dt;
  top_flux = ( yarea(0,1) * ( yvel0(0,1) + yvel0(1,1) +
                               yvel1(0,1) + yvel1(1,1) ) ) * 0.25 * dt;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change(0,0) = (volume(0,0))/(volume(0,0) + total_flux);

  //min_cell_volume = MIN( volume(0,0) + right_flux - left_flux + top_flux - bottom_flux ,
  //                         MIN(volume(0,0) + right_flux - left_flux,
  //                         volume(0,0) + top_flux - bottom_flux) );

  recip_volume = 1.0/volume(0,0);

  energy_change = ( pressure(0,0)/density0(0,0) +
                    viscosity(0,0)/density0(0,0) ) * total_flux * recip_volume;
  energy1(0,0) = energy0(0,0) - energy_change;
  density1(0,0) = density0(0,0) * volume_change(0,0);

}

#endif
