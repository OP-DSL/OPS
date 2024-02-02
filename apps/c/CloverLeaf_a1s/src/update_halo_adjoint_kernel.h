#ifndef UPDATE_HALO_ADJOINT_KERNEL_H
#define UPDATE_HALO_ADJOINT_KERNEL_H

//#include "data.h"


inline void update_halo_kernel1_b2_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(0,3) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(0,3) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(0,3) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(0,3) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(0,3) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(0,3) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_b1_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {

  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(0,1) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(0,1) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(0,1) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(0,1) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(0,1) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(0,1) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_t2_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(0,-3) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(0,-3) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(0,-3) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(0,-3) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(0,-3) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(0,-3) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_t1_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(0,-1) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(0,-1) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(0,-1) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(0,-1) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(0,-1) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(0,-1) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

//////////

inline void update_halo_kernel1_l2_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(3,0) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(3,0) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(3,0) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(3,0) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(3,0) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(3,0) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_l1_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(1,0) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(1,0) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(1,0) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(1,0) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(1,0) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(1,0) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_r2_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(-3,0) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(-3,0) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(-3,0) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(-3,0) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(-3,0) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(-3,0) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}

inline void update_halo_kernel1_r1_adjoint(ACC<double> &density0, ACC_A1S<double> &density0_a1s, ACC<double> &density1, ACC_A1S<double> &density1_a1s,
                          ACC<double> &energy0, ACC_A1S<double> &energy0_a1s, ACC<double> &energy1, ACC_A1S<double> &energy1_a1s,
                          ACC<double> &pressure, ACC_A1S<double> &pressure_a1s, ACC<double> &viscosity, ACC_A1S<double> &viscosity_a1s,
                          ACC<double> &soundspeed, const int* fields) {
  if(fields[FIELD_DENSITY0] == 1)  { density0_a1s(-1,0) += density0_a1s(0,0); density0_a1s(0,0) = 0;}
  if(fields[FIELD_DENSITY1] == 1)  { density1_a1s(-1,0) += density1_a1s(0,0); density1_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY0] == 1)  { energy0_a1s(-1,0) += energy0_a1s(0,0); energy0_a1s(0,0) = 0;}
  if(fields[FIELD_ENERGY1] == 1)  { energy1_a1s(-1,0) += energy1_a1s(0,0); energy1_a1s(0,0) = 0;}
  if(fields[FIELD_PRESSURE] == 1)  { pressure_a1s(-1,0) += pressure_a1s(0,0); pressure_a1s(0,0) = 0;}
  if(fields[FIELD_VISCOSITY] == 1)  { viscosity_a1s(-1,0) += viscosity_a1s(0,0); viscosity_a1s(0,0) = 0;}

}
////

inline void update_halo_kernel2_xvel_plus_4_a_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(0,4) += xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(0,4) += xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_plus_2_a_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(0,2) += xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(0,2) += xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_plus_4_b_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(0,-4) += xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(0,-4) += xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_plus_2_b_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(0,-2) += xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(0,-2) += xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

///

inline void update_halo_kernel2_xvel_minus_4_a_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(4,0) += -xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(4,0) += -xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_minus_2_a_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(2,0) += -xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(2,0) += -xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_minus_4_b_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(-4,0) += -xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(-4,0) += -xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_xvel_minus_2_b_adjoint(ACC<double> &xvel0, ACC_A1S<double> &xvel0_a1s, ACC<double> &xvel1, ACC_A1S<double> &xvel1_a1s, const int* fields)
{
  if(fields[FIELD_XVEL0] == 1)  { xvel0_a1s(-2,0) += -xvel0_a1s(0,0); xvel0_a1s(0,0) = 0;}
  if(fields[FIELD_XVEL1] == 1)  { xvel1_a1s(-2,0) += -xvel1_a1s(0,0); xvel1_a1s(0,0) = 0;}
}


///

inline void update_halo_kernel2_yvel_plus_4_a_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(4,0) += yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(4,0) += yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_plus_2_a_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(2,0) += yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(2,0) += yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_plus_4_b_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(-4,0) += yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(-4,0) += yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_plus_2_b_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(-2,0) += yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(-2,0) += yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

///

inline void update_halo_kernel2_yvel_minus_4_a_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(0,4) += -yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(0,4) += -yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_minus_2_a_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(0,2) += -yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(0,2) += -yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_minus_4_b_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(0,-4) += -yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(0,-4) += -yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}

inline void update_halo_kernel2_yvel_minus_2_b_adjoint(ACC<double> &yvel0, ACC_A1S<double> &yvel0_a1s, ACC<double> &yvel1, ACC_A1S<double> &yvel1_a1s, const int* fields) {
  if(fields[FIELD_YVEL0] == 1)  { yvel0_a1s(0,-2) += -yvel0_a1s(0,0); yvel0_a1s(0,0) = 0;}
  if(fields[FIELD_YVEL1] == 1)  { yvel1_a1s(0,-2) += -yvel1_a1s(0,0); yvel1_a1s(0,0) = 0;}
}


///

inline void update_halo_kernel3_plus_4_a_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(0,4) += vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(0,4) += mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_plus_2_a_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(0,2) += vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(0,2) += mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_plus_4_b_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(0,-4) += vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(0,-4) += mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_plus_2_b_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(0,-2) += vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(0,-2) += mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_minus_4_a_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(4,0) += -vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(4,0) += -mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_minus_2_a_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(2,0) += -vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(2,0) += -mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_minus_4_b_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(-4,0) += -vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(-4,0) += -mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}

inline void update_halo_kernel3_minus_2_b_adjoint(ACC<double> &vol_flux_x, ACC_A1S<double> &vol_flux_x_a1s, ACC<double> &mass_flux_x, ACC_A1S<double> &mass_flux_x_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_X] == 1)   { vol_flux_x_a1s(-2,0) += -vol_flux_x_a1s(0,0); vol_flux_x_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_X] == 1)  { mass_flux_x_a1s(-2,0) += -mass_flux_x_a1s(0,0); mass_flux_x_a1s(0,0) = 0;}
}


///


inline void update_halo_kernel4_plus_4_a_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(4,0) += vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(4,0) += mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_plus_2_a_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(2,0) += vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(2,0) += mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_plus_4_b_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(-4,0) += vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(-4,0) += mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_plus_2_b_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(-2,0) += vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(-2,0) += mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_minus_4_a_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(0,4) += -vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(0,4) += -mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_minus_2_a_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(0,2) += -vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(0,2) += -mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_minus_4_b_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(0,-4) += -vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(0,-4) += -mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

inline void update_halo_kernel4_minus_2_b_adjoint(ACC<double> &vol_flux_y, ACC_A1S<double> &vol_flux_y_a1s, ACC<double> &mass_flux_y, ACC_A1S<double> &mass_flux_y_a1s, const int* fields) {
  if(fields[FIELD_VOL_FLUX_Y] == 1)  { vol_flux_y_a1s(0,-2) += -vol_flux_y_a1s(0,0); vol_flux_y_a1s(0,0) = 0;}
  if(fields[FIELD_MASS_FLUX_Y] == 1)  { mass_flux_y_a1s(0,-2) += -mass_flux_y_a1s(0,0); mass_flux_y_a1s(0,0) = 0;}
}

#endif
