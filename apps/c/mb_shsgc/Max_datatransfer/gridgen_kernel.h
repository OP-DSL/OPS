#ifndef gridgen_kernel_H
#define gridgen_kernel_H

void gridgen_kernel(double *x, const int *id) {

  x[OPS_ACC0(0)] = xt +  id[0] *dx;
/*   if (x[OPS_ACC0(0)] >= -4.0){
     rho_new[OPS_ACC1(0)] = 1.0 + eps * sin(lambda *x[OPS_ACC0(0)]);
     rhou_new[OPS_ACC2(0)] = ur * rho_new[OPS_ACC1(0)];
     rhoE_new[OPS_ACC3(0)] = (pr / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
   }
   else {
     rho_new[OPS_ACC1(0)] = rhol;
     rhou_new[OPS_ACC2(0)] = ul2 * rho_new[OPS_ACC1(0)];
     rhoE_new[OPS_ACC3(0)] = (pl / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
   }
   rho_old[OPS_ACC5(0)]  = rho_new[OPS_ACC1(0)];
   rhou_old[OPS_ACC6(0)] = rhou_new[OPS_ACC2(0)];
   rhoE_old[OPS_ACC7(0)] = rhoE_new[OPS_ACC3(0)];

   rhoin[OPS_ACC4(0)] = rho_new[OPS_ACC1(0)];
   double *rho_new, double *rhou_new, double *rhoE_new,
   double* rhoin, double *rho_old, double *rhou_old,
   double *rhoE_old,, const int *idx
*/
}

#endif
