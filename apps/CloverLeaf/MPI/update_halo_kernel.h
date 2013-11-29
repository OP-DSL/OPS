#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

#include "data.h"


void update_halo_kernel1_b2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,3)];

}

void update_halo_kernel_b1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,1)];

}

void update_halo_kernel_t2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,-3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,-3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,-3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,-3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,-3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,-3)];

}

void update_halo_kernel_t1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(0,-1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(0,-1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(0,-1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(0,-1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(0,-1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(0,-1)];

}

//////////

void update_halo_kernel1_l2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(3,0)];

}

void update_halo_kernel1_l1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(1,0)];

}

void update_halo_kernel1_r2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(-3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(-3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(-3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(-3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(-3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(-3,0)];

}

void update_halo_kernel1_r1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed ) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0)] = density1[OPS_ACC0(-1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0)] = energy0[OPS_ACC0(-1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0)] = energy1[OPS_ACC0(-1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0)] = pressure[OPS_ACC0(-1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0)] = viscosity[OPS_ACC0(-1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0)] = soundspeed[OPS_ACC0(-1,0)];

}
////

void update_halo_kernel2_xvel_plus_4_a(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,4)];
}
void update_halo_kernel2_xvel_plus_2_a(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,2)];
}

void update_halo_kernel2_xvel_plus_4_b(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,-4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,-4)];
}
void update_halo_kernel2_xvel_plus_2_b(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = xvel0[OPS_ACC0(0,-2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = xvel1[OPS_ACC1(0,-2)];
}

///

void update_halo_kernel2_xvel_minus_4_a(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(4,0)];
}
void update_halo_kernel2_xvel_minus_2_a(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(2,0)];
}

void update_halo_kernel2_xvel_minus_4_b(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(-4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(-4,0)];
}
void update_halo_kernel2_xvel_minus_2_b(double *xvel0, double *xvel1){
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0)] = -xvel0[OPS_ACC0(-2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0)] = -xvel1[OPS_ACC1(-2,0)];
}


///

void update_halo_kernel2_yvel_plus_4_a(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(4,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(4,0)];
}
void update_halo_kernel2_yvel_plus_2_a(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(2,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(2,0)];
}

void update_halo_kernel2_yvel_plus_4_b(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(-4,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(-4,0)];
}
void update_halo_kernel2_yvel_plus_2_b(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = yvel0[OPS_ACC0(-2,0)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = yvel1[OPS_ACC1(-2,0)];
}

///

void update_halo_kernel2_yvel_minus_4_a(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,4)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,4)];
}
void update_halo_kernel2_yvel_minus_2_a(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,2)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,2)];
}

void update_halo_kernel2_yvel_minus_4_b(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,-4)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,-4)];
}
void update_halo_kernel2_yvel_minus_2_b(double *yvel0, double *yvel1){
  if(fields[FIELD_XVEL0] == 1) yvel0[OPS_ACC0(0,0)] = -yvel0[OPS_ACC0(0,-2)];
  if(fields[FIELD_XVEL1] == 1) yvel1[OPS_ACC1(0,0)] = -yvel1[OPS_ACC1(0,-2)];
}


///

void update_halo_kernel3_plus_4_a(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = vol_flux_x[OPS_ACC0(0,4)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = mass_flux_x[OPS_ACC1(0,4)];
}
void update_halo_kernel3_plus_2_a(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = vol_flux_x[OPS_ACC0(0,2)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = mass_flux_x[OPS_ACC1(0,2)];
}

void update_halo_kernel3_plus_4_b(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = vol_flux_x[OPS_ACC0(0,-4)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = mass_flux_x[OPS_ACC1(0,-4)];
}
void update_halo_kernel3_plus_2_b(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = vol_flux_x[OPS_ACC0(0,-2)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = mass_flux_x[OPS_ACC1(0,-2)];
}

void update_halo_kernel3_minus_4_a(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = -(vol_flux_x[OPS_ACC0(4,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = -(mass_flux_x[OPS_ACC1(4,0)]);
}
void update_halo_kernel3_minus_2_a(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = -(vol_flux_x[OPS_ACC0(2,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = -(mass_flux_x[OPS_ACC1(2,0)]);
}
void update_halo_kernel3_minus_4_b(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = -(vol_flux_x[OPS_ACC0(-4,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = -(mass_flux_x[OPS_ACC1(-4,0)]);
}
void update_halo_kernel3_minus_2_b(double *vol_flux_x, double *mass_flux_x){
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0)]  = -(vol_flux_x[OPS_ACC0(-2,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0)] = -(mass_flux_x[OPS_ACC1(-2,0)]);
}


///


void update_halo_kernel4_plus_4_a(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = vol_flux_y[OPS_ACC0(4,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = mass_flux_y[OPS_ACC1(4,0)];
}
void update_halo_kernel4_plus_2_a(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = vol_flux_y[OPS_ACC0(2,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = mass_flux_y[OPS_ACC1(2,0)];
}
void update_halo_kernel4_plus_4_b(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = vol_flux_y[OPS_ACC0(-4,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = mass_flux_y[OPS_ACC1(-4,0)];
}
void update_halo_kernel4_plus_2_b(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = vol_flux_y[OPS_ACC0(-2,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = mass_flux_y[OPS_ACC1(-2,0)];
}

void update_halo_kernel4_minus_4_a(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = -(vol_flux_y[OPS_ACC0(0,4)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = -(mass_flux_y[OPS_ACC1(0,4)]);
}
void update_halo_kernel4_minus_2_a(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = -(vol_flux_y[OPS_ACC0(0,2)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = -(mass_flux_y[OPS_ACC1(0,2)]);
}
void update_halo_kernel4_minus_4_b(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = -(vol_flux_y[OPS_ACC0(0,-4)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = -(mass_flux_y[OPS_ACC1(0,-4)]);
}
void update_halo_kernel4_minus_2_b(double *vol_flux_y, double *mass_flux_y){
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0)] = -(vol_flux_y[OPS_ACC0(0,-2)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0)] = -(mass_flux_y[OPS_ACC1(0,-2)]);
}


#endif
