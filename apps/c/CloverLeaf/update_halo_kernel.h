#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

//#include "data.h"


inline void update_halo_kernel1_b2(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(0,3);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(0,3);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,3);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,3);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(0,3);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(0,3);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(0,3);

}

inline void update_halo_kernel1_b1(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {

  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(0,1);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(0,1);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,1);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,1);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(0,1);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(0,1);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(0,1);

}

inline void update_halo_kernel1_t2(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(0,-3);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(0,-3);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,-3);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,-3);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(0,-3);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(0,-3);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(0,-3);

}

inline void update_halo_kernel1_t1(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(0,-1);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(0,-1);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(0,-1);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(0,-1);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(0,-1);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(0,-1);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(0,-1);

}

//////////

inline void update_halo_kernel1_l2(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(3,0);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(3,0);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(3,0);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(3,0);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(3,0);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(3,0);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(3,0);

}

inline void update_halo_kernel1_l1(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(1,0);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(1,0);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(1,0);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(1,0);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(1,0);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(1,0);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(1,0);

}

inline void update_halo_kernel1_r2(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(-3,0);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(-3,0);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(-3,0);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(-3,0);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(-3,0);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(-3,0);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(-3,0);

}

inline void update_halo_kernel1_r1(ACC<double> &density0, ACC<double> &density1,
                          ACC<double> &energy0, ACC<double> &energy1,
                          ACC<double> &pressure, ACC<double> &viscosity,
                          ACC<double> &soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0(0,0) = density0(-1,0);
  if(fields[FIELD_DENSITY1] == 1) density1(0,0) = density1(-1,0);
  if(fields[FIELD_ENERGY0] == 1) energy0(0,0) = energy0(-1,0);
  if(fields[FIELD_ENERGY1] == 1) energy1(0,0) = energy1(-1,0);
  if(fields[FIELD_PRESSURE] == 1) pressure(0,0) = pressure(-1,0);
  if(fields[FIELD_VISCOSITY] == 1) viscosity(0,0) = viscosity(-1,0);
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed(0,0) = soundspeed(-1,0);

}
////

inline void update_halo_kernel2_xvel_plus_4_a(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = xvel0(0,4);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = xvel1(0,4);
}

inline void update_halo_kernel2_xvel_plus_2_a(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = xvel0(0,2);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = xvel1(0,2);
}

inline void update_halo_kernel2_xvel_plus_4_b(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = xvel0(0,-4);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = xvel1(0,-4);
}

inline void update_halo_kernel2_xvel_plus_2_b(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = xvel0(0,-2);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = xvel1(0,-2);
}

///

inline void update_halo_kernel2_xvel_minus_4_a(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = -xvel0(4,0);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = -xvel1(4,0);
}

inline void update_halo_kernel2_xvel_minus_2_a(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = -xvel0(2,0);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = -xvel1(2,0);
}

inline void update_halo_kernel2_xvel_minus_4_b(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = -xvel0(-4,0);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = -xvel1(-4,0);
}

inline void update_halo_kernel2_xvel_minus_2_b(ACC<double> &xvel0, ACC<double> &xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0(0,0) = -xvel0(-2,0);
  if(fields[FIELD_XVEL1] == 1) xvel1(0,0) = -xvel1(-2,0);
}


///

inline void update_halo_kernel2_yvel_plus_4_a(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = yvel0(4,0);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = yvel1(4,0);
}

inline void update_halo_kernel2_yvel_plus_2_a(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = yvel0(2,0);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = yvel1(2,0);
}

inline void update_halo_kernel2_yvel_plus_4_b(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = yvel0(-4,0);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = yvel1(-4,0);
}

inline void update_halo_kernel2_yvel_plus_2_b(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = yvel0(-2,0);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = yvel1(-2,0);
}

///

inline void update_halo_kernel2_yvel_minus_4_a(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = -yvel0(0,4);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = -yvel1(0,4);
}

inline void update_halo_kernel2_yvel_minus_2_a(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = -yvel0(0,2);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = -yvel1(0,2);
}

inline void update_halo_kernel2_yvel_minus_4_b(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = -yvel0(0,-4);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = -yvel1(0,-4);
}

inline void update_halo_kernel2_yvel_minus_2_b(ACC<double> &yvel0, ACC<double> &yvel1, const int* fields) {
  if(fields[FIELD_YVEL0] == 1) yvel0(0,0) = -yvel0(0,-2);
  if(fields[FIELD_YVEL1] == 1) yvel1(0,0) = -yvel1(0,-2);
}


///

inline void update_halo_kernel3_plus_4_a(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = vol_flux_x(0,4);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = mass_flux_x(0,4);
}

inline void update_halo_kernel3_plus_2_a(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = vol_flux_x(0,2);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = mass_flux_x(0,2);
}

inline void update_halo_kernel3_plus_4_b(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = vol_flux_x(0,-4);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = mass_flux_x(0,-4);
}

inline void update_halo_kernel3_plus_2_b(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = vol_flux_x(0,-2);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = mass_flux_x(0,-2);
}

inline void update_halo_kernel3_minus_4_a(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = -(vol_flux_x(4,0));
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = -(mass_flux_x(4,0));
}

inline void update_halo_kernel3_minus_2_a(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = -(vol_flux_x(2,0));
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = -(mass_flux_x(2,0));
}

inline void update_halo_kernel3_minus_4_b(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = -(vol_flux_x(-4,0));
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = -(mass_flux_x(-4,0));
}

inline void update_halo_kernel3_minus_2_b(ACC<double> &vol_flux_x, ACC<double> &mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x(0,0)  = -(vol_flux_x(-2,0));
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x(0,0) = -(mass_flux_x(-2,0));
}


///


inline void update_halo_kernel4_plus_4_a(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = vol_flux_y(4,0);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = mass_flux_y(4,0);
}

inline void update_halo_kernel4_plus_2_a(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = vol_flux_y(2,0);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = mass_flux_y(2,0);
}

inline void update_halo_kernel4_plus_4_b(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = vol_flux_y(-4,0);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = mass_flux_y(-4,0);
}

inline void update_halo_kernel4_plus_2_b(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = vol_flux_y(-2,0);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = mass_flux_y(-2,0);
}

inline void update_halo_kernel4_minus_4_a(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = -(vol_flux_y(0,4));
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = -(mass_flux_y(0,4));
}

inline void update_halo_kernel4_minus_2_a(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = -(vol_flux_y(0,2));
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = -(mass_flux_y(0,2));
}

inline void update_halo_kernel4_minus_4_b(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = -(vol_flux_y(0,-4));
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = -(mass_flux_y(0,-4));
}

inline void update_halo_kernel4_minus_2_b(ACC<double> &vol_flux_y, ACC<double> &mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y(0,0) = -(vol_flux_y(0,-2));
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y(0,0) = -(mass_flux_y(0,-2));
}

#endif
