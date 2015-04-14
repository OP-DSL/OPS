#ifndef init_kernel_H
#define init_kernel_H


void init_kernel(const double *x,double *rho_new, double *rhou_new, double *rhoE_new,
                       double* rhoin, double *rho_old, double *rhou_old,
                       double *rhoE_old) {
  if (x[OPS_ACC0(0)] >= -4.0){
    rho_new[OPS_ACC1(0)] = 1.0 + eps * sin(lambda *x[OPS_ACC0(0)]);
    rhou_new[OPS_ACC2(0)] = ur * rho_new[OPS_ACC1(0)];
    rhoE_new[OPS_ACC3(0)] = (pr / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
  }
  else {
    rho_new[OPS_ACC1(0)] = rhol;
    rhou_new[OPS_ACC2(0)] = ul * rho_new[OPS_ACC1(0)];
    rhoE_new[OPS_ACC3(0)] = (pl / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
  }
  rho_old[OPS_ACC5(0)]  = rho_new[OPS_ACC1(0)];
  rhou_old[OPS_ACC6(0)] = rhou_new[OPS_ACC2(0)];
  rhoE_old[OPS_ACC7(0)] = rhoE_new[OPS_ACC3(0)];

  rhoin[OPS_ACC4(0)] = rho_new[OPS_ACC1(0)];

}

#endif
