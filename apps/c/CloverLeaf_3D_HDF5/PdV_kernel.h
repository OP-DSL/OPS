#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

#include "data.h"
#include "definitions.h"

void PdV_kernel_predict(const double *xarea, const double *xvel0,
                const double *yarea, const double *yvel0,
                double *volume_change, const double *volume,
                const double *pressure,
                const double *density0, double *density1,
                const double *viscosity,
                const double *energy0, double *energy1, const double *zarea, const double *zvel0) {

  double recip_volume, energy_change;
  double right_flux, left_flux, top_flux, bottom_flux, back_flux, front_flux, total_flux;

  left_flux = ( xarea[OPS_ACC0(0,0,0)] * ( xvel0[OPS_ACC1(0,0,0)] + xvel0[OPS_ACC1(0,1,0)] +
                                           xvel0[OPS_ACC1(0,0,1)] + xvel0[OPS_ACC1(0,1,1)] +
                                           xvel0[OPS_ACC1(0,0,0)] + xvel0[OPS_ACC1(0,1,0)] +
                                           xvel0[OPS_ACC1(0,0,1)] + xvel0[OPS_ACC1(0,1,1)] ) ) * 0.125 * dt * 0.5;
  right_flux = ( xarea[OPS_ACC0(1,0,0)] * ( xvel0[OPS_ACC1(1,0,0)] + xvel0[OPS_ACC1(1,1,0)] +
                                            xvel0[OPS_ACC1(1,0,1)] + xvel0[OPS_ACC1(1,1,1)] +
                                            xvel0[OPS_ACC1(1,0,0)] + xvel0[OPS_ACC1(1,1,0)] +
                                            xvel0[OPS_ACC1(1,0,1)] + xvel0[OPS_ACC1(1,1,1)] ) ) * 0.125 * dt * 0.5;

  bottom_flux = ( yarea[OPS_ACC2(0,0,0)] * ( yvel0[OPS_ACC3(0,0,0)] + yvel0[OPS_ACC3(1,0,0)] +
                                             yvel0[OPS_ACC3(0,0,1)] + yvel0[OPS_ACC3(1,0,1)] +
                                             yvel0[OPS_ACC3(0,0,0)] + yvel0[OPS_ACC3(1,0,0)] +
                                             yvel0[OPS_ACC3(0,0,1)] + yvel0[OPS_ACC3(1,0,1)] ) ) * 0.125* dt * 0.5;
  top_flux = ( yarea[OPS_ACC2(0,1,0)] * ( yvel0[OPS_ACC3(0,1,0)] + yvel0[OPS_ACC3(1,1,0)] +
                                          yvel0[OPS_ACC3(0,1,1)] + yvel0[OPS_ACC3(1,1,1)] +
                                          yvel0[OPS_ACC3(0,1,0)] + yvel0[OPS_ACC3(1,1,0)] +
                                          yvel0[OPS_ACC3(0,1,1)] + yvel0[OPS_ACC3(1,1,1)] ) ) * 0.125 * dt * 0.5;

  back_flux = ( zarea[OPS_ACC12(0,0,0)] * ( zvel0[OPS_ACC13(0,0,0)] + zvel0[OPS_ACC13(1,0,0)] +
                                            zvel0[OPS_ACC13(0,1,0)] + zvel0[OPS_ACC13(1,1,0)] +
                                            zvel0[OPS_ACC13(0,0,0)] + zvel0[OPS_ACC13(1,0,0)] +
                                            zvel0[OPS_ACC13(0,1,0)] + zvel0[OPS_ACC13(1,1,0)] ) ) * 0.125* dt * 0.5;
  front_flux = ( zarea[OPS_ACC12(0,0,1)] * ( zvel0[OPS_ACC13(0,0,1)] + zvel0[OPS_ACC13(1,0,1)] +
                                             zvel0[OPS_ACC13(0,1,1)] + zvel0[OPS_ACC13(1,1,1)] +
                                             zvel0[OPS_ACC13(0,0,1)] + zvel0[OPS_ACC13(1,0,1)] +
                                             zvel0[OPS_ACC13(0,1,1)] + zvel0[OPS_ACC13(1,1,1)] ) ) * 0.125 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux + front_flux - back_flux;

  volume_change[OPS_ACC4(0,0,0)] = (volume[OPS_ACC5(0,0,0)])/(volume[OPS_ACC5(0,0,0)] + total_flux);
  recip_volume = 1.0/volume[OPS_ACC5(0,0,0)];
  energy_change = ( pressure[OPS_ACC6(0,0,0)]/density0[OPS_ACC7(0,0,0)] +
                    viscosity[OPS_ACC9(0,0,0)]/density0[OPS_ACC7(0,0,0)] ) * total_flux * recip_volume;
  energy1[OPS_ACC11(0,0,0)] = energy0[OPS_ACC10(0,0,0)] - energy_change;
  density1[OPS_ACC8(0,0,0)] = density0[OPS_ACC7(0,0,0)] * volume_change[OPS_ACC4(0,0,0)];

}

void PdV_kernel_nopredict(const double *xarea, const double *xvel0, const double *xvel1,
                const double *yarea, const double *yvel0, const double *yvel1,
                double *volume_change, const double *volume,
                const double *pressure,
                const double *density0, double *density1,
                const double *viscosity,
                const double *energy0, double *energy1, const double *zarea, const double *zvel0, const double *zvel1) { //14,15,16

  double recip_volume, energy_change;
  double right_flux, left_flux, top_flux, bottom_flux, back_flux, front_flux, total_flux;

  left_flux = ( xarea[OPS_ACC0(0,0,0)] * ( xvel0[OPS_ACC1(0,0,0)] + xvel0[OPS_ACC1(0,1,0)] +
                                           xvel0[OPS_ACC1(0,0,1)] + xvel0[OPS_ACC1(0,1,1)] +
                                           xvel1[OPS_ACC2(0,0,0)] + xvel1[OPS_ACC2(0,1,0)] +
                                           xvel1[OPS_ACC2(0,0,1)] + xvel1[OPS_ACC2(0,1,1)] ) ) * 0.125 * dt;
  right_flux = ( xarea[OPS_ACC0(1,0,0)] * ( xvel0[OPS_ACC1(1,0,0)] + xvel0[OPS_ACC1(1,1,0)] +
                                            xvel0[OPS_ACC1(1,0,1)] + xvel0[OPS_ACC1(1,1,1)] +
                                            xvel1[OPS_ACC2(1,0,0)] + xvel1[OPS_ACC2(1,1,0)] +
                                            xvel1[OPS_ACC2(1,0,1)] + xvel1[OPS_ACC2(1,1,1)] ) ) * 0.125 * dt;

  bottom_flux = ( yarea[OPS_ACC3(0,0,0)] * ( yvel0[OPS_ACC4(0,0,0)] + yvel0[OPS_ACC4(1,0,0)] +
                                             yvel0[OPS_ACC4(0,0,1)] + yvel0[OPS_ACC4(1,0,1)] +
                                             yvel1[OPS_ACC5(0,0,0)] + yvel1[OPS_ACC5(1,0,0)] +
                                             yvel1[OPS_ACC5(0,0,1)] + yvel1[OPS_ACC5(1,0,1)] ) ) * 0.125* dt;
  top_flux = ( yarea[OPS_ACC3(0,1,0)] * ( yvel0[OPS_ACC4(0,1,0)] + yvel0[OPS_ACC4(1,1,0)] +
                                          yvel0[OPS_ACC4(0,1,1)] + yvel0[OPS_ACC4(1,1,1)] +
                                          yvel1[OPS_ACC5(0,1,0)] + yvel1[OPS_ACC5(1,1,0)] +
                                          yvel1[OPS_ACC5(0,1,1)] + yvel1[OPS_ACC5(1,1,1)]) ) * 0.125 * dt;

  back_flux = ( zarea[OPS_ACC14(0,0,0)] * ( zvel0[OPS_ACC15(0,0,0)] + zvel0[OPS_ACC15(1,0,0)] +
                                            zvel0[OPS_ACC15(0,1,0)] + zvel0[OPS_ACC15(1,1,0)] +
                                            zvel1[OPS_ACC16(0,0,0)] + zvel1[OPS_ACC16(1,0,0)] +
                                            zvel1[OPS_ACC16(0,1,0)] + zvel1[OPS_ACC16(1,1,0)] ) ) * 0.125* dt;
  front_flux = ( zarea[OPS_ACC14(0,0,1)] * ( zvel0[OPS_ACC15(0,0,1)] + zvel0[OPS_ACC15(1,0,1)] +
                                             zvel0[OPS_ACC15(0,1,1)] + zvel0[OPS_ACC15(1,1,1)] +
                                             zvel1[OPS_ACC16(0,0,1)] + zvel1[OPS_ACC16(1,0,1)] +
                                             zvel1[OPS_ACC16(0,1,1)] + zvel1[OPS_ACC16(1,1,1)]) ) * 0.125 * dt;

  total_flux = right_flux - left_flux + top_flux - bottom_flux + front_flux - back_flux;

  volume_change[OPS_ACC6(0,0,0)] = (volume[OPS_ACC7(0,0,0)])/(volume[OPS_ACC7(0,0,0)] + total_flux);
  recip_volume = 1.0/volume[OPS_ACC7(0,0,0)];
  energy_change = ( pressure[OPS_ACC8(0,0,0)]/density0[OPS_ACC9(0,0,0)] +
                    viscosity[OPS_ACC11(0,0,0)]/density0[OPS_ACC9(0,0,0)] ) * total_flux * recip_volume;
  energy1[OPS_ACC13(0,0,0)] = energy0[OPS_ACC12(0,0,0)] - energy_change;
  density1[OPS_ACC10(0,0,0)] = density0[OPS_ACC9(0,0,0)] * volume_change[OPS_ACC6(0,0,0)];

}
#endif
