#ifndef RIEMANN_KERNEL_H
#define RIEMANN_KERNEL_H

#include "vars.h"


void Riemann_kernel(double* rho_new, double *rhou_new,  double* rhoE_new, 
                    double* alam, double* r, double* al) {
  
  double rl, rr, rho, leftu, rightu, u, hl, hr, h, Vsq, csq, c, g;
  double dw1, dw2, dw3, delpc2, rdeluc;
  double ri[3][3];

  rl = sqrt(rho_new[OPS_ACC0(0)]);
  rr = sqrt(rho_new[OPS_ACC0(1)]);
  rho = rl + rr;
  u = ((rhou_new[OPS_ACC1(0)] / rl) + (rhou_new[OPS_ACC1(1)] / rr)) / rho ;
  double fni = rhou_new[OPS_ACC1(0)] * rhou_new[OPS_ACC1(0)] / rho_new[OPS_ACC0(0)] ;
  double p = gam1 * (rhoE_new[OPS_ACC2(0)] - 0.5 * fni);
  hl = (rhoE_new[OPS_ACC2(0)] + p)  / rl ;
  fni = rhou_new[OPS_ACC1(1)] * rhou_new[OPS_ACC1(1)] / rho_new[OPS_ACC0(1)] ;
  p = gam1 * (rhoE_new[OPS_ACC2(1)] - 0.5 * fni);
  hr = (rhoE_new[OPS_ACC2(1)] + p)  / rr ;
  h = (hl + hr)/rho;
  Vsq = u*u;
  csq = gam1 * (h - 0.5 * Vsq);
  g = gam1 / csq;
  c = sqrt(csq);
  
  alam[OPS_ACC_MD3(0,0)] = u - c;
  alam[OPS_ACC_MD3(1,0)] = u;
  alam[OPS_ACC_MD3(2,0)] = u + c;

  r[OPS_ACC_MD4(0,0)] = 1.0;
  r[OPS_ACC_MD4(1,0)] = 1.0;
  r[OPS_ACC_MD4(2,0)] = 1.0;

  r[OPS_ACC_MD4(3,0)] = u - c;
  r[OPS_ACC_MD4(4,0)] = u;
  r[OPS_ACC_MD4(5,0)] = u + c;

  r[OPS_ACC_MD4(6,0)] = h - u * c;
  r[OPS_ACC_MD4(7,0)] = 0.5 * Vsq;
  r[OPS_ACC_MD4(8,0)] = h + u * c;

  for (int m=0; m<9; m++)
    r[OPS_ACC_MD4(m,0)] = r[OPS_ACC_MD4(m,0)] / csq;
    
  dw1 = rho_new[OPS_ACC0(1)] - rho_new[OPS_ACC0(0)]; 
  dw2 = rhou_new[OPS_ACC1(1)] - rhou_new[OPS_ACC1(0)]; 
  dw3 = rhoE_new[OPS_ACC2(1)] - rhoE_new[OPS_ACC2(0)]; 		
  
  delpc2 = gam1 * ( dw3 + 0.50 * Vsq * dw1  - u * dw2) / csq;
  rdeluc = ( dw2 - u * dw1) / c ;
  
  al[OPS_ACC_MD5(0,0)] = 0.5 * (delpc2 - rdeluc);
  al[OPS_ACC_MD5(1,0)] = dw1 - delpc2 ;
  al[OPS_ACC_MD5(2,0)] = 0.5 * ( delpc2 + rdeluc );
  
  for (int m=0; m<3; m++) 
    al[OPS_ACC_MD5(m,0)] = al[OPS_ACC_MD5(m,0)] * csq;	
}

#endif