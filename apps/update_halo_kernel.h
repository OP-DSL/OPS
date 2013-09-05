#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

#include "data.h"

void update_halo_kernel( double **density0, double **density1,
                         double **energy0, double **energy1,
                         double **pressure,double **viscosity,
                         double **soundspeed) {

  if(fields[FIELD_DENSITY0] == 1) *density0[0] = *density0[1];
  if(fields[FIELD_DENSITY1] == 1) *density1[0] = *density1[1];
  if(fields[FIELD_ENERGY0] == 1) *energy0[0] = *energy0[1];
  if(fields[FIELD_ENERGY1] == 1) *energy1[0] = *energy1[1];
  if(fields[FIELD_PRESSURE] == 1) *pressure[0] = *pressure[1];
  if(fields[FIELD_VISCOSITY] == 1) *viscosity[0] = *viscosity[1];
  if(fields[FIELD_SOUNDSPEED] == 1) *soundspeed[0] = *soundspeed[1];
}



void update_halo_kernel2( double **xvel0, double **xvel1,
                         double **yvel0, double **yvel1,
                         double **vol_flux_x,double **vol_flux_y,
                         double **mass_flux_x, double **mass_flux_y) {

  if(fields[FIELD_XVEL0] == 1) *xvel0[0] = *xvel0[1];
  if(fields[FIELD_XVEL1] == 1) *xvel1[0] = *xvel1[1];
  if(fields[FIELD_YVEL0] == 1) *yvel0[0] = *yvel0[1];
  if(fields[FIELD_YVEL1] == 1) *yvel1[0] = *yvel1[1];
  if(fields[FIELD_VOL_FLUX_X] == 1) *vol_flux_x[0] = *vol_flux_x[1];
  if(fields[FIELD_VOL_FLUX_Y] == 1) *vol_flux_y[0] = *vol_flux_y[1];
  if(fields[FIELD_MASS_FLUX_X] == 1) *mass_flux_x[0] = *mass_flux_x[1];
  if(fields[FIELD_MASS_FLUX_Y] == 1) *mass_flux_y[0] = *mass_flux_y[1];
}

#endif
