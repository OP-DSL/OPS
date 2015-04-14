#ifndef UPDATERK3_KERNEL_H
#define UPDATERK3_KERNEL_H

void updateRK3_kernel(double *rho_new, double* rhou_new, double* rhoE_new,
                      double *rho_old, double* rhou_old, double* rhoE_old,
                      double *rho_res, double *rhou_res, double *rhoE_res,
                      const double* a1, const double* a2) {

  rho_new[OPS_ACC0(0)]  = rho_old[OPS_ACC3(0)] + dt * a1[0] * (-rho_res[OPS_ACC6(0)]);
  rhou_new[OPS_ACC1(0)] = rhou_old[OPS_ACC4(0)] + dt * a1[0] * (-rhou_res[OPS_ACC7(0)]);
  rhoE_new[OPS_ACC2(0)] = rhoE_old[OPS_ACC5(0)] + dt * a1[0] * (-rhoE_res[OPS_ACC8(0)]);
  // update old state
  rho_old[OPS_ACC3(0)]  = rho_old[OPS_ACC3(0)] + dt * a2[0] * (-rho_res[OPS_ACC6(0)]);
  rhou_old[OPS_ACC4(0)] = rhou_old[OPS_ACC4(0)] + dt * a2[0] * (-rhou_res[OPS_ACC7(0)]);
  rhoE_old[OPS_ACC5(0)] = rhoE_old[OPS_ACC5(0)] + dt * a2[0] * (-rhoE_res[OPS_ACC8(0)]);
  rho_res[OPS_ACC6(0)]  = 0;
  rhou_res[OPS_ACC7(0)] = 0;
  rhoE_res[OPS_ACC8(0)] = 0;
  }

#endif


#ifndef convective_kernel_H
#define convective_kernel_H

void convective_kernel(const double *rho_new, const double *rhou_new, const double *rhoE_new,
                       double *rho_res, double *rhou_res, double *rhoE_res) {
  // THE VARS ARE 0, 1, 2, 3, 4, 5
  double dxi, c1d, b1d, a1d;
  c1d = 1.f/12.f;
  b1d = 2.f/3.f;
  a1d = 0.0f;
  dxi = 1.0f/dx;
  // derivative of rhou for conservation
  rho_res[OPS_ACC3(0)]  += dxi * (c1d * (rhou_new[OPS_ACC1(-2)] - rhou_new[OPS_ACC1(2)])
                        +  b1d * (rhou_new[OPS_ACC1(1)] - rhou_new[OPS_ACC1(-1)] ));
  // momentum conservation is derivative of rhouu
  rhou_res[OPS_ACC4(0)] += dxi * (c1d * (pow(rhou_new[OPS_ACC1(-2)],2) / rho_new[OPS_ACC0(-2)]
                        -                pow(rhou_new[OPS_ACC1(2)],2) / rho_new[OPS_ACC0(2)])
                        +         b1d * (pow(rhou_new[OPS_ACC1(1)],2) / rho_new[OPS_ACC0(1)]
                        -                pow(rhou_new[OPS_ACC1(-1)],2) / rho_new[OPS_ACC0(-1)]));
  // add derivative of P to the momentum equation
  // p = gam1*(rhoE - 0.5* rhou^2/rho)
  rhou_res[OPS_ACC4(0)] += dxi * (c1d * (gam1 * (rhoE_new[OPS_ACC2(-2)] - 0.5f * pow(rhou_new[OPS_ACC1(-2)],2) / rho_new[OPS_ACC0(-2)])
                        -                gam1 * (rhoE_new[OPS_ACC2(2)]  - 0.5f * pow(rhou_new[OPS_ACC1(2)],2)  / rho_new[OPS_ACC0(2)]))
                        +         b1d * (gam1 * (rhoE_new[OPS_ACC2(1)]  - 0.5f * pow(rhou_new[OPS_ACC1(1)],2)  / rho_new[OPS_ACC0(1)])
                        -                gam1 * (rhoE_new[OPS_ACC2(-1)] - 0.5f * pow(rhou_new[OPS_ACC1(-1)],2) / rho_new[OPS_ACC0(-1)])));
  // Now the energy equation derivative
  // first (p+rhoE)*u derivative is written as (gam * rhoE - (gam-1) * 0.5 ru^2/rho) * (rhou/rho)
  rhoE_res[OPS_ACC5(0)] += dxi * (c1d * ( (rhou_new[OPS_ACC1(-2)]/ rho_new[OPS_ACC0(-2)]) * (gam * rhoE_new[OPS_ACC2(-2)] - gam1 * 0.5f * pow(rhou_new[OPS_ACC1(-2)],2) / rho_new[OPS_ACC0(-2)])
                        -                 (rhou_new[OPS_ACC1(2)] / rho_new[OPS_ACC0(2)] ) * (gam * rhoE_new[OPS_ACC2(2)]  - gam1 * 0.5f * pow(rhou_new[OPS_ACC1(2)],2)  / rho_new[OPS_ACC0(2)]))
                        +         b1d * ( (rhou_new[OPS_ACC1(1)] / rho_new[OPS_ACC0(1)] ) * (gam * rhoE_new[OPS_ACC2(1)]  - gam1 * 0.5f * pow(rhou_new[OPS_ACC1(1)],2)  / rho_new[OPS_ACC0(1)])
                        -                 (rhou_new[OPS_ACC1(-1)]/ rho_new[OPS_ACC0(-1)]) * (gam * rhoE_new[OPS_ACC2(-1)] - gam1 * 0.5f * pow(rhou_new[OPS_ACC1(-1)],2) / rho_new[OPS_ACC0(-1)])));

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

#ifndef checkop_kernel_H
#define checkop_kernel_H

void checkop_kernel(const double *rho_new, const double *x, const double *rhoin, double *pre, double *post,
  int *num) {
  /*calculate error post shock, for this i need data of total time for the simulation
  assuming total time is constant i.e 9005 iterations*/
  double diff;
  diff = (rho_new[OPS_ACC0(0)] - rhoin[OPS_ACC2(0)]);
  if(fabs(diff)<0.01 && x[OPS_ACC1(0)] > -4.1){
    *post = *post + diff*diff;
    *num = *num + 1;
  }
  else
    *pre = *pre + (rho_new[OPS_ACC0(0)] - rhol)* (rho_new[OPS_ACC0(0)] - rhol);
}
#endif