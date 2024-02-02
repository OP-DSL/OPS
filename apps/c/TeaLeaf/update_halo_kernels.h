#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

//#include "data.h"


inline void update_halo_kernel1_b2(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd, const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(0,3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(0,3)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(0,3)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(0,3)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(0,3)];

}

inline void update_halo_kernel1_b1(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {

  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(0,1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(0,1)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(0,1)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(0,1)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(0,1)];

}

inline void update_halo_kernel1_t2(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-3)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(0,-3)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(0,-3)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(0,-3)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(0,-3)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(0,-3)];

}

inline void update_halo_kernel1_t1(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(0,-1)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(0,-1)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(0,-1)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(0,-1)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(0,-1)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(0,-1)];

}

//////////

inline void update_halo_kernel1_l2(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(3,0)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(3,0)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(3,0)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(3,0)];

}

inline void update_halo_kernel1_l1(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(1,0)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(1,0)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(1,0)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(1,0)];

}

inline void update_halo_kernel1_r2(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-3,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(-3,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(-3,0)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(-3,0)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(-3,0)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(-3,0)];

}

inline void update_halo_kernel1_r1(double *density0,
                          double *energy0, double *energy1,
                          double *u, double *p,
                          double *sd , const int* fields) {
  if(fields[FIELD_DENSITY] == 1) density0[OPS_ACC0(0,0)] = density0[OPS_ACC0(-1,0)];
  if(fields[FIELD_ENERGY0] == 1) energy0[OPS_ACC1(0,0)] = energy0[OPS_ACC1(-1,0)];
  if(fields[FIELD_ENERGY1] == 1) energy1[OPS_ACC2(0,0)] = energy1[OPS_ACC2(-1,0)];
  if(fields[FIELD_U] == 1) u[OPS_ACC3(0,0)] = u[OPS_ACC3(-1,0)];
  if(fields[FIELD_P] == 1) p[OPS_ACC4(0,0)] = p[OPS_ACC4(-1,0)];
  if(fields[FIELD_SD] == 1) sd[OPS_ACC5(0,0)] = sd[OPS_ACC5(-1,0)];

}

#endif
