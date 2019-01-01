#ifndef RIEMANN_KERNEL_H
#define RIEMANN_KERNEL_H

#include "vars.h"


void Riemann_kernel(const ACC<double>& rho_new, const ACC<double> &rhou_new, const ACC<double>& rhoE_new,
                    ACC<double>& alam, ACC<double>& r, ACC<double>& al) {

  double rl, rr, rho, u, hl, hr, h, Vsq, csq, c;
  double dw1, dw2, dw3, delpc2, rdeluc;


  rl = sqrt(rho_new(0));
  rr = sqrt(rho_new(1));
  rho = rl + rr;
  u = ((rhou_new(0) / rl) + (rhou_new(1) / rr)) / rho ;
  double fni = rhou_new(0) * rhou_new(0) / rho_new(0) ;
  double p = gam1 * (rhoE_new(0) - 0.5 * fni);
  hl = (rhoE_new(0) + p)  / rl ;
  fni = rhou_new(1) * rhou_new(1) / rho_new(1) ;
  p = gam1 * (rhoE_new(1) - 0.5 * fni);
  hr = (rhoE_new(1) + p)  / rr ;
  h = (hl + hr)/rho;
  Vsq = u*u;
  csq = gam1 * (h - 0.5 * Vsq);
  c = sqrt(csq);

  alam(0,0) = u - c;
  alam(1,0) = u;
  alam(2,0) = u + c;

  r(0,0) = 1.0;
  r(1,0) = 1.0;
  r(2,0) = 1.0;

  r(3,0) = u - c;
  r(4,0) = u;
  r(5,0) = u + c;

  r(6,0) = h - u * c;
  r(7,0) = 0.5 * Vsq;
  r(8,0) = h + u * c;

  for (int m=0; m<9; m++)
    r(m,0) = r(m,0) / csq;

  dw1 = rho_new(1) - rho_new(0);
  dw2 = rhou_new(1) - rhou_new(0);
  dw3 = rhoE_new(1) - rhoE_new(0);

  delpc2 = gam1 * ( dw3 + 0.50 * Vsq * dw1  - u * dw2) / csq;
  rdeluc = ( dw2 - u * dw1) / c ;

  al(0,0) = 0.5 * (delpc2 - rdeluc);
  al(1,0) = dw1 - delpc2 ;
  al(2,0) = 0.5 * ( delpc2 + rdeluc );

  for (int m=0; m<3; m++)
    al(m,0) = al(m,0) * csq;
}
#endif
