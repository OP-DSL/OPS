#include "vars.h"

#ifndef xder1_kernel_H
#define xder1_kernel_H

void xder1_kernel(const double *inp, double *out) {
  double dlx = 1/(12.00*dx);
  out[OPS_ACC1(0)] = (inp[OPS_ACC0(-2)] - inp[OPS_ACC0(2)]  + 8.0 *(
  inp[OPS_ACC0(1)] - inp[OPS_ACC0(-1)] )) * dlx;
}

#endif


#ifndef UPDATERK3_KERNEL_H
#define UPDATERK3_KERNEL_H

void updateRK3_kernel(double *rho_new, double* rhou_new, double* rhoE_new,
                      double *rho_old, double* rhou_old, double* rhoE_old,
                      double *rho_res, double *rhou_res, double *rhoE_res,
                      const double* a1, const double* a2) {

  rho_new[OPS_ACC0(0)] = rho_old[OPS_ACC3(0)] + dt * a1[0] * (-rho_res[OPS_ACC6(0)]);
  rhou_new[OPS_ACC1(0)] = rhou_old[OPS_ACC4(0)] + dt * a1[0] * (-rhou_res[OPS_ACC7(0)]);
  rhoE_new[OPS_ACC2(0)] = rhoE_old[OPS_ACC5(0)] + dt * a1[0] * (-rhoE_res[OPS_ACC8(0)]);
  // update old state
  rho_old[OPS_ACC3(0)] = rho_old[OPS_ACC3(0)] + dt * a2[0] * (-rho_res[OPS_ACC6(0)]);
  rhou_old[OPS_ACC4(0)] = rhou_old[OPS_ACC4(0)] + dt * a2[0] * (-rhou_res[OPS_ACC7(0)]);
  rhoE_old[OPS_ACC5(0)] = rhoE_old[OPS_ACC5(0)] + dt * a2[0] * (-rhoE_res[OPS_ACC8(0)]);
  rho_res[OPS_ACC6(0)] = 0;
  rhou_res[OPS_ACC7(0)] = 0;
  rhoE_res[OPS_ACC8(0)] = 0;
  }

#endif


#ifndef SAVE_KERNEL_H
#define SAVE_KERNEL_H

void save_kernel(double *rho_old, double *rhou_old, double *rhoE_old,
                 const double *rho_new, const double *rhou_new, const double *rhoE_new) {
  rho_old[OPS_ACC0(0)]=rho_new[OPS_ACC3(0)];
  rhou_old[OPS_ACC1(0)]=rhou_new[OPS_ACC4(0)];
  rhoE_old[OPS_ACC2(0)]=rhoE_new[OPS_ACC5(0)];
  }

#endif

#ifndef residue_eval_H
#define residue_eval_H


void residue_eval(const double *der1, const double *der2, const double *der3,
                  double *rho_res, double *rhou_res, double *rhoE_res) {
  rho_res[OPS_ACC3(0)] = der1[OPS_ACC0(0)];
  rhou_res[OPS_ACC4(0)] = der2[OPS_ACC1(0)];
  rhoE_res[OPS_ACC5(0)] = der3[OPS_ACC2(0)];
  }
#endif


#ifndef calvar_kernel_H
#define calvar_kernel_H

void calvar_kernel(const double *rho_new, const double *rhou_new, const double *rhoE_new,
                       double *workarray2, double *workarray3) {
  double p, rhoi, u;
  rhoi = 1/rho_new[OPS_ACC0(0)];
  u = rhou_new[OPS_ACC1(0)] * rhoi;
  p = gam1 * (rhoE_new[OPS_ACC2(0)] - 0.5 * rho_new[OPS_ACC0(0)]* u * u);
  // cal p+rhouu
  workarray2[OPS_ACC3(0)] = p + rhou_new[OPS_ACC1(0)] * u ;
  workarray3[OPS_ACC4(0)] = (p + rhoE_new[OPS_ACC2(0)]) * u ;
  }
#endif