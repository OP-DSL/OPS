#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

//#include "data.h"


inline void update_halo_kernel1_b2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,3,0)];

}

inline void update_halo_kernel1_b1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {

  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,1,0)];

}

inline void update_halo_kernel1_t2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,-3,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,-3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,-3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,-3,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,-3,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,-3,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,-3,0)];

}

inline void update_halo_kernel1_t1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,-1,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,-1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,-1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,-1,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,-1,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,-1,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,-1,0)];

}

//////////

inline void update_halo_kernel1_l2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(3,0,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(3,0,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(3,0,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(3,0,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(3,0,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(3,0,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(3,0,0)];

}

inline void update_halo_kernel1_l1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(1,0,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(1,0,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(1,0,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(1,0,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(1,0,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(1,0,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(1,0,0)];

}

inline void update_halo_kernel1_r2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(-3,0,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(-3,0,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(-3,0,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(-3,0,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(-3,0,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(-3,0,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(-3,0,0)];

}

inline void update_halo_kernel1_r1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(-1,0,0)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(-1,0,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(-1,0,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(-1,0,0)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(-1,0,0)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(-1,0,0)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(-1,0,0)];

}
////

inline void update_halo_kernel1_ba2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,0,3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,0,3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,0,3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,0,3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,0,3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,0,3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,0,3)];

}

inline void update_halo_kernel1_ba1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {

  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,0,1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,0,1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,0,1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,0,1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,0,1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,0,1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,0,1)];

}

inline void update_halo_kernel1_fr2(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,0,-3)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,0,-3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,0,-3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,0,-3)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,0,-3)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,0,-3)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,0,-3)];

}

inline void update_halo_kernel1_fr1(double *density0, double *density1,
                          double *energy0, double *energy1,
                          double *pressure, double *viscosity,
                          double *soundspeed , const int* fields) {
  if(fields[FIELD_DENSITY0] == 1) density0[OPS_ACC0(0,0,0)] = density0[OPS_ACC0(0,0,-1)];
  if(fields[FIELD_DENSITY1] == 1) density1[OPS_ACC1(0,0,0)] = density1[OPS_ACC1(0,0,-1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC2(0,0,0)] = energy0[OPS_ACC2(0,0,-1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC3(0,0,0)] = energy1[OPS_ACC3(0,0,-1)];
  if(fields[FIELD_PRESSURE] == 1) pressure[OPS_ACC4(0,0,0)] = pressure[OPS_ACC4(0,0,-1)];
  if(fields[FIELD_VISCOSITY] == 1) viscosity[OPS_ACC5(0,0,0)] = viscosity[OPS_ACC5(0,0,-1)];
  if(fields[FIELD_SOUNDSPEED] == 1) soundspeed[OPS_ACC6(0,0,0)] = soundspeed[OPS_ACC6(0,0,-1)];

}

/////// xvel begin
inline void update_halo_kernel2_xvel_plus_4_bot(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,4,0)];
}

inline void update_halo_kernel2_xvel_plus_2_bot(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,2,0)];
}

inline void update_halo_kernel2_xvel_plus_4_top(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,-4,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,-4,0)];
}

inline void update_halo_kernel2_xvel_plus_2_top(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,-2,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,-2,0)];
}

///

inline void update_halo_kernel2_xvel_minus_4_left(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = -xvel0[OPS_ACC0(4,0,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = -xvel1[OPS_ACC1(4,0,0)];
}

inline void update_halo_kernel2_xvel_minus_2_left(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = -xvel0[OPS_ACC0(2,0,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = -xvel1[OPS_ACC1(2,0,0)];
}

inline void update_halo_kernel2_xvel_minus_4_right(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = -xvel0[OPS_ACC0(-4,0,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = -xvel1[OPS_ACC1(-4,0,0)];
}

inline void update_halo_kernel2_xvel_minus_2_right(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = -xvel0[OPS_ACC0(-2,0,0)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = -xvel1[OPS_ACC1(-2,0,0)];
}

///

inline void update_halo_kernel2_xvel_plus_4_back(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,0,4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel2_xvel_plus_2_back(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,0,2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel2_xvel_plus_4_front(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel2_xvel_plus_2_front(double *xvel0, double *xvel1, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1) xvel0[OPS_ACC0(0,0,0)] = xvel0[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_XVEL1] == 1) xvel1[OPS_ACC1(0,0,0)] = xvel1[OPS_ACC1(0,0,-2)];
}

/// xvel end

/////// yvel begin
inline void update_halo_kernel2_yvel_minus_4_bot(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = -yvel0[OPS_ACC0(0,4,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = -yvel1[OPS_ACC1(0,4,0)];
}

inline void update_halo_kernel2_yvel_minus_2_bot(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = -yvel0[OPS_ACC0(0,2,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = -yvel1[OPS_ACC1(0,2,0)];
}

inline void update_halo_kernel2_yvel_minus_4_top(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = -yvel0[OPS_ACC0(0,-4,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = -yvel1[OPS_ACC1(0,-4,0)];
}

inline void update_halo_kernel2_yvel_minus_2_top(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = -yvel0[OPS_ACC0(0,-2,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = -yvel1[OPS_ACC1(0,-2,0)];
}

///

inline void update_halo_kernel2_yvel_plus_4_left(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(4,0,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(4,0,0)];
}

inline void update_halo_kernel2_yvel_plus_2_left(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(2,0,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(2,0,0)];
}

inline void update_halo_kernel2_yvel_plus_4_right(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(-4,0,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(-4,0,0)];
}

inline void update_halo_kernel2_yvel_plus_2_right(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(-2,0,0)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(-2,0,0)];
}


///

inline void update_halo_kernel2_yvel_plus_4_back(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(0,0,4)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel2_yvel_plus_2_back(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(0,0,2)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel2_yvel_plus_4_front(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel2_yvel_plus_2_front(double *yvel0, double *yvel1, const int* fields)
{
  if(fields[FIELD_YVEL0] == 1) yvel0[OPS_ACC0(0,0,0)] = yvel0[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_YVEL1] == 1) yvel1[OPS_ACC1(0,0,0)] = yvel1[OPS_ACC1(0,0,-2)];
}


/// yvel end


/////// zvel begin
inline void update_halo_kernel2_zvel_plus_4_bot(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(0,4,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(0,4,0)];
}

inline void update_halo_kernel2_zvel_plus_2_bot(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(0,2,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(0,2,0)];
}

inline void update_halo_kernel2_zvel_plus_4_top(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(0,-4,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(0,-4,0)];
}

inline void update_halo_kernel2_zvel_plus_2_top(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(0,-2,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(0,-2,0)];
}

///

inline void update_halo_kernel2_zvel_plus_4_left(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(4,0,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(4,0,0)];
}

inline void update_halo_kernel2_zvel_plus_2_left(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(2,0,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(2,0,0)];
}

inline void update_halo_kernel2_zvel_plus_4_right(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(-4,0,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(-4,0,0)];
}

inline void update_halo_kernel2_zvel_plus_2_right(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = zvel0[OPS_ACC0(-2,0,0)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = zvel1[OPS_ACC1(-2,0,0)];
}


///

inline void update_halo_kernel2_zvel_minus_4_back(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = -zvel0[OPS_ACC0(0,0,4)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = -zvel1[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel2_zvel_minus_2_back(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = -zvel0[OPS_ACC0(0,0,2)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = -zvel1[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel2_zvel_minus_4_front(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = -zvel0[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = -zvel1[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel2_zvel_minus_2_front(double *zvel0, double *zvel1, const int* fields)
{
  if(fields[FIELD_ZVEL0] == 1) zvel0[OPS_ACC0(0,0,0)] = -zvel0[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_ZVEL1] == 1) zvel1[OPS_ACC1(0,0,0)] = -zvel1[OPS_ACC1(0,0,-2)];
}


/// zvel end


inline void update_halo_kernel3_plus_4_a(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,4,0)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,4,0)];
}

inline void update_halo_kernel3_plus_2_a(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,2,0)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,2,0)];
}

inline void update_halo_kernel3_plus_4_b(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,-4,0)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,-4,0)];
}

inline void update_halo_kernel3_plus_2_b(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,-2,0)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,-2,0)];
}

inline void update_halo_kernel3_minus_4_a(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = -(vol_flux_x[OPS_ACC0(4,0,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = -(mass_flux_x[OPS_ACC1(4,0,0)]);
}

inline void update_halo_kernel3_minus_2_a(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = -(vol_flux_x[OPS_ACC0(2,0,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = -(mass_flux_x[OPS_ACC1(2,0,0)]);
}

inline void update_halo_kernel3_minus_4_b(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = -(vol_flux_x[OPS_ACC0(-4,0,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = -(mass_flux_x[OPS_ACC1(-4,0,0)]);
}

inline void update_halo_kernel3_minus_2_b(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = -(vol_flux_x[OPS_ACC0(-2,0,0)]);
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = -(mass_flux_x[OPS_ACC1(-2,0,0)]);
}

inline void update_halo_kernel3_plus_4_back(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,0,4)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel3_plus_2_back(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,0,2)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel3_plus_4_front(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel3_plus_2_front(double *vol_flux_x, double *mass_flux_x, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)  vol_flux_x[OPS_ACC0(0,0,0)]  = vol_flux_x[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_MASS_FLUX_X] == 1) mass_flux_x[OPS_ACC1(0,0,0)] = mass_flux_x[OPS_ACC1(0,0,-2)];
}


///

inline void update_halo_kernel4_minus_4_a(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = -(vol_flux_y[OPS_ACC0(0,4,0)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = -(mass_flux_y[OPS_ACC1(0,4,0)]);
}

inline void update_halo_kernel4_minus_2_a(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = -(vol_flux_y[OPS_ACC0(0,2,0)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = -(mass_flux_y[OPS_ACC1(0,2,0)]);
}

inline void update_halo_kernel4_minus_4_b(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = -(vol_flux_y[OPS_ACC0(0,-4,0)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = -(mass_flux_y[OPS_ACC1(0,-4,0)]);
}

inline void update_halo_kernel4_minus_2_b(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = -(vol_flux_y[OPS_ACC0(0,-2,0)]);
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = -(mass_flux_y[OPS_ACC1(0,-2,0)]);
}

inline void update_halo_kernel4_plus_4_a(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = vol_flux_y[OPS_ACC0(4,0,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(4,0,0)];
}

inline void update_halo_kernel4_plus_2_a(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = vol_flux_y[OPS_ACC0(2,0,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(2,0,0)];
}

inline void update_halo_kernel4_plus_4_b(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = vol_flux_y[OPS_ACC0(-4,0,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(-4,0,0)];
}

inline void update_halo_kernel4_plus_2_b(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1) vol_flux_y[OPS_ACC0(0,0,0)] = vol_flux_y[OPS_ACC0(-2,0,0)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(-2,0,0)];
}

inline void update_halo_kernel4_plus_4_back(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  vol_flux_y[OPS_ACC0(0,0,0)]  = vol_flux_y[OPS_ACC0(0,0,4)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel4_plus_2_back(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  vol_flux_y[OPS_ACC0(0,0,0)]  = vol_flux_y[OPS_ACC0(0,0,2)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel4_plus_4_front(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  vol_flux_y[OPS_ACC0(0,0,0)]  = vol_flux_y[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel4_plus_2_front(double *vol_flux_y, double *mass_flux_y, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  vol_flux_y[OPS_ACC0(0,0,0)]  = vol_flux_y[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_MASS_FLUX_Y] == 1) mass_flux_y[OPS_ACC1(0,0,0)] = mass_flux_y[OPS_ACC1(0,0,-2)];
}

///


inline void update_halo_kernel5_plus_4_a(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = vol_flux_z[OPS_ACC0(0,4,0)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = mass_flux_z[OPS_ACC1(0,4,0)];
}

inline void update_halo_kernel5_plus_2_a(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = vol_flux_z[OPS_ACC0(0,2,0)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = mass_flux_z[OPS_ACC1(0,2,0)];
}

inline void update_halo_kernel5_plus_4_b(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = vol_flux_z[OPS_ACC0(0,-4,0)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = mass_flux_z[OPS_ACC1(0,-4,0)];
}

inline void update_halo_kernel5_plus_2_b(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = vol_flux_z[OPS_ACC0(0,-2,0)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = mass_flux_z[OPS_ACC1(0,-2,0)];
}

inline void update_halo_kernel5_plus_4_left(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = (vol_flux_z[OPS_ACC0(4,0,0)]);
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = (mass_flux_z[OPS_ACC1(4,0,0)]);
}

inline void update_halo_kernel5_plus_2_left(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = (vol_flux_z[OPS_ACC0(2,0,0)]);
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = (mass_flux_z[OPS_ACC1(2,0,0)]);
}

inline void update_halo_kernel5_plus_4_right(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = (vol_flux_z[OPS_ACC0(-4,0,0)]);
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = (mass_flux_z[OPS_ACC1(-4,0,0)]);
}

inline void update_halo_kernel5_plus_2_right(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1) vol_flux_z[OPS_ACC0(0,0,0)] = (vol_flux_z[OPS_ACC0(-2,0,0)]);
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = (mass_flux_z[OPS_ACC1(-2,0,0)]);
}

inline void update_halo_kernel5_minus_4_back(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1)  vol_flux_z[OPS_ACC0(0,0,0)]  = -vol_flux_z[OPS_ACC0(0,0,4)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = -mass_flux_z[OPS_ACC1(0,0,4)];
}

inline void update_halo_kernel5_minus_2_back(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1)  vol_flux_z[OPS_ACC0(0,0,0)]  = -vol_flux_z[OPS_ACC0(0,0,2)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = -mass_flux_z[OPS_ACC1(0,0,2)];
}

inline void update_halo_kernel5_minus_4_front(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1)  vol_flux_z[OPS_ACC0(0,0,0)]  = -vol_flux_z[OPS_ACC0(0,0,-4)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = -mass_flux_z[OPS_ACC1(0,0,-4)];
}

inline void update_halo_kernel5_minus_2_front(double *vol_flux_z, double *mass_flux_z, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Z] == 1)  vol_flux_z[OPS_ACC0(0,0,0)]  = -vol_flux_z[OPS_ACC0(0,0,-2)];
  if(fields[FIELD_MASS_FLUX_Z] == 1) mass_flux_z[OPS_ACC1(0,0,0)] = -mass_flux_z[OPS_ACC1(0,0,-2)];
}

#endif
