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
  /* variables for evaluating derivatives, velocity is a vector and is a 3D array
   f irst index is the component o*f vector, secon*d index is the component of first
   index in i,j,k directions. Third index 0,1,2,3,4 are the indices at -2,-1,0,1,2
   locations of the component of velocity in the direction represented by second
   index, these are used to evaluate 4th order central difference scheme, if the
   scheme needs to be changed then change the last index, similarly for pressure
   the first index is pressure along i,j,k directions and second index is same as
   the third index for velocity */
  int nd =1;
  double vel[nd][nd][5], p[nd][5], rho[nd][5], rhou[nd][nd][5], rhoe[nd][5];
  double delta[nd], momresidue[nd], energyresidue, rhoresidue;
  double vel_x[nd][nd], p_x[nd], rhou_x[nd][nd], rhouu_x[nd][nd],rhoe_x[nd];
  delta[0] = 1.0/(12.00*dx);
  double dxi, c1d, b1d, a1d;

  for (int i=0;i<nd;i++){
    momresidue[i] = 0.0;
    energyresidue = 0.0;
    rhoresidue    = 0.0;
  }
  // save variables into temporary arrays for easy evaluation of derivatives
  for (int k=0; k<5; k++){
    vel[0][0][k] = rhou_new[OPS_ACC1(k-2)]/rho_new[OPS_ACC0(k-2)] ;
    rho[0][k] = rho_new[OPS_ACC0(k-2)];
    rhou[0][0][k] = rhou_new[OPS_ACC1(k-2)];
    rhoe[0][k] = rhoE_new[OPS_ACC2(k-2)];
  }
  /* caluclate the dependent variables, pressure and temperature and viscocity */
  for (int j=0; j<nd; j++){
    for (int k=0; k<5; k++){
      p[j][k] = gam1 * (rhoe[j][k] - 0.5 * rho[j][k]*(pow(vel[0][j][k],2))
      // TODO need to find a good solution for 3D for finding total velocity
      ); // the velocity evaluation should be changed
    }
  }
  c1d = 1.f/12.f;
  b1d = 2.f/3.f;
  a1d = 0.0f;
  dxi = 1.0f/dx;
  /* caluclate the derivatives here */
  for (int i=0; i<nd;i++){
    for (int j=0; j<nd; j++){
      vel_x[i][j] = delta[j] * (vel[i][j][0] - vel[i][j][4] + 8.0 *
      (vel[i][j][3] - vel[i][j][1]));
      rhou_x[i][j] = delta[j] * (rhou[i][j][0] - rhou[i][j][4] + 8.0 *
      (rhou[i][j][3] - rhou[i][j][1]));
      rhouu_x[i][j] = delta[j] * (rhou[i][j][0] * vel[j][j][0] -
      rhou[i][j][4] * vel[j][j][4] + 8.0 * (rhou[i][j][3] * vel[j][j][3]
      - rhou[i][j][1] * vel[j][j][1]));
    }
    p_x[i] = delta[i] * (p[i][0] - p[i][4] + 8.0 * (p[i][3] - p[i][1]));
    //  find the energy derivative below (rhoE+p)*u finished on 12/03/15
    rhoe_x[i] = delta[i] * ((rhoe[i][0]+p[i][0])*vel[i][i][0] -
    (rhoe[i][4]+p[i][4])*vel[i][i][4]  + 8.0 * ((rhoe[i][3]+p[i][3])*vel[i][i][3] -
    (rhoe[i][1]+p[i][1])*vel[i][i][1]));
  }
  //  convective flux terms arrangement finished on 12/03/15
  for (int i=0; i<nd;i++){
    rhoresidue +=  rhou_x[i][i]; // Continuity
    for (int j=0; j<nd; j++){
      momresidue[i]  +=  (rhouu_x[i][j]);
    }
    momresidue[i]  += p_x[i];
    energyresidue  +=  rhoe_x[i];
  }
  // viscous flux terms arrangement


  // equate fluxes to residue
  rho_res[OPS_ACC3(0)]  = rhoresidue;
  rhou_res[OPS_ACC4(0)] = momresidue[0];
  rhoE_res[OPS_ACC5(0)] = energyresidue;

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