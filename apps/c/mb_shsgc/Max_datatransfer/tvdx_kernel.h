#ifndef tvdx_kernel_H
#define tvdx_kernel_H


// this contains riemann, limiter, secondorder dissipation and update kernels
void Riemann_kernel(const double* rho_new, const double *rhou_new, const double* rhoE_new,
                    double* alam, double* r, double* al) {
  double rl, rr, rho, u, hl, hr, h, Vsq, csq, c;
  double dw1, dw2, dw3, delpc2, rdeluc;
//   double ri[3][3];

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

void limiter_kernel(const double* al, double *tht, double* gt) {

  double aalm, aal, all, ar, gtt;
  for (int m=0; m < 3 ;m++) {
    aalm = fabs(al[OPS_ACC_MD0(m,-1)]);
    aal = fabs(al[OPS_ACC_MD0(m,0)]);
    tht[OPS_ACC_MD1(m,0)] = fabs (aal - aalm) / (aal + aalm + del2);
    all = al[OPS_ACC_MD0(m,-1)];
    ar = al[OPS_ACC_MD0(m,0)];
    gtt = all * ( ar * ar + del2 ) + ar * (all * all + del2);
    gt[OPS_ACC_MD2(m,0)]= gtt / (ar * ar + all * all + 2.00 * del2);
  }
}
void tvd_kernel(const double *tht, double* ep2) {
  double maxim;
  for (int m=0; m < 3 ;m++) {
    if (tht[OPS_ACC_MD0(m,0)] > tht[OPS_ACC_MD0(m,1)])
      maxim = tht[OPS_ACC_MD0(m,0)];
    else
      maxim = tht[OPS_ACC_MD0(m,1)];
    ep2[OPS_ACC_MD1(m,0)] = akap2 * maxim;
  }
}

void fact_kernel(const double* eff, double *s) {
  double fact;
  for (int m=0; m < 3 ;m++) {
    fact  = 0.50 * dt / dx ;
    s[OPS_ACC_MD1(m,0)] = -fact * (eff[OPS_ACC_MD0(m,0)] - eff[OPS_ACC_MD0(m,-1)]);
  }
}

void vars_kernel(const double* alam, const double* al, const double *gt,
                 double* cmp,  double* cf) {
  double  anu, aaa, ga, qf, ww;
  for (int m=0; m < 3 ;m++) {
    anu = alam[OPS_ACC_MD0(m,0)];
    aaa = al[OPS_ACC_MD1(m,0)];
    ga = aaa * ( gt[OPS_ACC_MD2(m,1)] - gt[OPS_ACC_MD2(m,0)]) / (pow(aaa,2.0) + del2);
    qf = sqrt ( con + pow(anu,2.0));
    cmp[OPS_ACC_MD3(m,0)] = 0.50 * qf;
    ww = anu + cmp[OPS_ACC_MD3(m,0)] * ga;
    qf = sqrt(con + pow(ww,2.0));
    cf[OPS_ACC_MD4(m,0)] = qf;
  }
}

void calupwindeff_kernel(const double* cmp, const double *gt, const double* cf,
          const double* al, const double* ep2, const double* r, double* eff) {
  double e1 = (cmp[OPS_ACC_MD0(0,0)] * (gt[OPS_ACC_MD1(0,0)] + gt[OPS_ACC_MD1(0,1)])
  - cf[OPS_ACC_MD2(0,0)] * al[OPS_ACC_MD3(0,0)]) * ep2[OPS_ACC_MD4(0,0)];
  double e2 = (cmp[OPS_ACC_MD0(1,0)] * (gt[OPS_ACC_MD1(1,0)] + gt[OPS_ACC_MD1(1,1)])
  - cf[OPS_ACC_MD2(1,0)] * al[OPS_ACC_MD3(1,0)]) * ep2[OPS_ACC_MD4(1,0)];
  double e3 = (cmp[OPS_ACC_MD0(2,0)] * (gt[OPS_ACC_MD1(2,0)] + gt[OPS_ACC_MD1(2,1)])
  - cf[OPS_ACC_MD2(2,0)] * al[OPS_ACC_MD3(2,0)]) * ep2[OPS_ACC_MD4(2,0)];

  eff[OPS_ACC_MD6(0,0)]=e1 * r[OPS_ACC_MD5(0,0)] + e2 * r[OPS_ACC_MD5(1,0)] + e3 * r[OPS_ACC_MD5(2,0)];
  eff[OPS_ACC_MD6(1,0)]=e1 * r[OPS_ACC_MD5(3,0)] + e2 * r[OPS_ACC_MD5(4,0)] + e3 * r[OPS_ACC_MD5(5,0)];
  eff[OPS_ACC_MD6(2,0)]=e1 * r[OPS_ACC_MD5(6,0)] + e2 * r[OPS_ACC_MD5(7,0)] + e3 * r[OPS_ACC_MD5(8,0)];
}

void update_kernel(double *rho_new, double *rhou_new, double *rhoE_new, const double *s) {
  rho_new[OPS_ACC0(0)]  = rho_new[OPS_ACC0(0)]  + s[OPS_ACC_MD3(0,0)];
  rhou_new[OPS_ACC1(0)] = rhou_new[OPS_ACC1(0)] + s[OPS_ACC_MD3(1,0)];
  rhoE_new[OPS_ACC2(0)] = rhoE_new[OPS_ACC2(0)] + s[OPS_ACC_MD3(2,0)];
}
#endif
