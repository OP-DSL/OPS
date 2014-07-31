#ifndef VISCOSITY_KERNEL_H
#define VISCOSITY_KERNEL_H


void viscosity_kernel( const double *xvel0, const double *yvel0,
                       const double *celldx, const double *celldy,
                       const double *pressure, const double *density0,
                       double *viscosity, const double *zvel0, const double *celldz, const double *xarea, const double *yarea, const double *zarea) { //7,8,9,10,11

  double grad2,
         pgradx,pgrady,pgradz,
         pgradx2,pgrady2,pgradz2,
         grad,
         ygrad, xgrad, zgrad,
         div,
         strain2,
         limiter,
         pgrad;

  //int s2D_00_P10_0P1_P1P1[]  = {0,0, 1,0, 0,1, 1,1};
  double ugradx1=xvel0[OPS_ACC0(0,0,0)]+xvel0[OPS_ACC0(0,1,0)]+xvel0[OPS_ACC0(0,0,1)]+xvel0[OPS_ACC0(0,1,1)];
  double ugradx2=xvel0[OPS_ACC0(1,0,0)]+xvel0[OPS_ACC0(1,1,0)]+xvel0[OPS_ACC0(1,0,1)]+xvel0[OPS_ACC0(1,1,1)];
  double ugrady1=xvel0[OPS_ACC0(0,0,0)]+xvel0[OPS_ACC0(1,0,0)]+xvel0[OPS_ACC0(0,0,1)]+xvel0[OPS_ACC0(1,0,1)];
  double ugrady2=xvel0[OPS_ACC0(0,1,0)]+xvel0[OPS_ACC0(1,1,0)]+xvel0[OPS_ACC0(0,1,1)]+xvel0[OPS_ACC0(1,1,1)];
  double ugradz1=xvel0[OPS_ACC0(0,0,0)]+xvel0[OPS_ACC0(1,0,0)]+xvel0[OPS_ACC0(0,1,0)]+xvel0[OPS_ACC0(1,1,0)];
  double ugradz2=xvel0[OPS_ACC0(0,0,1)]+xvel0[OPS_ACC0(1,0,1)]+xvel0[OPS_ACC0(0,1,1)]+xvel0[OPS_ACC0(1,1,1)];

  double vgradx1=yvel0[OPS_ACC1(0,0,0)]+yvel0[OPS_ACC1(0,1,0)]+yvel0[OPS_ACC1(0,0,1)]+yvel0[OPS_ACC1(0,1,1)];
  double vgradx2=yvel0[OPS_ACC1(1,0,0)]+yvel0[OPS_ACC1(1,1,0)]+yvel0[OPS_ACC1(1,0,1)]+yvel0[OPS_ACC1(1,1,1)];
  double vgrady1=yvel0[OPS_ACC1(0,0,0)]+yvel0[OPS_ACC1(1,0,0)]+yvel0[OPS_ACC1(0,0,1)]+yvel0[OPS_ACC1(1,0,1)];
  double vgrady2=yvel0[OPS_ACC1(0,1,0)]+yvel0[OPS_ACC1(1,1,0)]+yvel0[OPS_ACC1(0,1,1)]+yvel0[OPS_ACC1(1,1,1)];
  double vgradz1=yvel0[OPS_ACC1(0,0,0)]+yvel0[OPS_ACC1(1,0,0)]+yvel0[OPS_ACC1(0,1,0)]+yvel0[OPS_ACC1(1,1,0)];
  double vgradz2=yvel0[OPS_ACC1(0,0,1)]+yvel0[OPS_ACC1(1,0,1)]+yvel0[OPS_ACC1(0,1,1)]+yvel0[OPS_ACC1(1,1,1)];

  double wgradx1=zvel0[OPS_ACC7(0,0,0)]+zvel0[OPS_ACC7(0,1,0)]+zvel0[OPS_ACC7(0,0,1)]+zvel0[OPS_ACC7(0,1,1)];
  double wgradx2=zvel0[OPS_ACC7(1,0,0)]+zvel0[OPS_ACC7(1,1,0)]+zvel0[OPS_ACC7(1,0,1)]+zvel0[OPS_ACC7(1,1,1)];
  double wgrady1=zvel0[OPS_ACC7(0,0,0)]+zvel0[OPS_ACC7(1,0,0)]+zvel0[OPS_ACC7(0,0,1)]+zvel0[OPS_ACC7(1,0,1)];
  double wgrady2=zvel0[OPS_ACC7(0,1,0)]+zvel0[OPS_ACC7(1,1,0)]+zvel0[OPS_ACC7(0,1,1)]+zvel0[OPS_ACC7(1,1,1)];
  double wgradz1=zvel0[OPS_ACC7(0,0,0)]+zvel0[OPS_ACC7(1,0,0)]+zvel0[OPS_ACC7(0,1,0)]+zvel0[OPS_ACC7(1,1,0)];
  double wgradz2=zvel0[OPS_ACC7(0,0,1)]+zvel0[OPS_ACC7(1,0,1)]+zvel0[OPS_ACC7(0,1,1)]+zvel0[OPS_ACC7(1,1,1)];

  div = xarea[OPS_ACC9(0,0,0)]*(ugradx2-ugradx1) + yarea[OPS_ACC10(0,0,0)]*(vgrady2-vgrady1) + zarea[OPS_ACC11(0,0,0)]*(wgradz2-wgradz1);

  double xx = 0.25*(ugradx2-ugradx1)/(celldx[OPS_ACC2(0,0,0)]);
  double yy = 0.25*(vgrady2-vgrady1)/(celldy[OPS_ACC3(0,0,0)]);
  double zz = 0.25*(wgradz2-wgradz1)/(celldz[OPS_ACC8(0,0,0)]);
  double xy = 0.25*(ugrady2-ugrady1)/(celldy[OPS_ACC3(0,0,0)])+0.25*(vgradx2-vgradx1)/(celldx[OPS_ACC2(0,0,0)]);
  double xz = 0.25*(ugradz2-ugradz1)/(celldz[OPS_ACC8(0,0,0)])+0.25*(wgradx2-wgradx1)/(celldx[OPS_ACC2(0,0,0)]);
  double yz = 0.25*(vgradz2-vgradz1)/(celldz[OPS_ACC8(0,0,0)])+0.25*(wgrady2-wgrady1)/(celldy[OPS_ACC3(0,0,0)]);


  //int s2D_10_M10_01_0M1[]  = {1,0, -1,0, 0,1, 0,-1};
  pgradx = (pressure[OPS_ACC4(1,0,0)] - pressure[OPS_ACC4(-1,0,0)])/(celldx[OPS_ACC2(0,0,0)]+ celldx[OPS_ACC2(1,0,0)]);
  pgrady = (pressure[OPS_ACC4(0,1,0)] - pressure[OPS_ACC4(0,-1,0)])/(celldy[OPS_ACC3(0,0,0)]+ celldy[OPS_ACC3(0,1,0)]);
  pgradz = (pressure[OPS_ACC4(0,0,1)] - pressure[OPS_ACC4(0,0,-1)])/(celldz[OPS_ACC8(0,0,0)]+ celldz[OPS_ACC8(0,0,1)]);

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;
  pgradz2 = pgradz * pgradz;
  limiter = (xx*pgradx2+yy*pgrady2+zz*pgradz2
          +  xy*pgradx*pgrady+xz*pgradx*pgradz+yz*pgrady*pgradz)
                / MAX(pgradx2+pgrady2+pgradz2,1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        viscosity[OPS_ACC6(0,0,0)] = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, fabs(pgrady)), pgrady);
    pgradz = SIGN( MAX(1.0e-16, fabs(pgradz)), pgradz);
    pgrad = sqrt(pgradx*pgradx + pgrady*pgrady + pgradz*pgradz);
    xgrad = fabs(celldx[OPS_ACC2(0,0,0)] * pgrad/pgradx);
    ygrad = fabs(celldy[OPS_ACC3(0,0,0)] * pgrad/pgrady);
    zgrad = fabs(celldz[OPS_ACC8(0,0,0)] * pgrad/pgradz);
    grad  = MIN(xgrad,MIN(ygrad,zgrad));
    grad2 = grad*grad;

    viscosity[OPS_ACC6(0,0,0)] = 2.0 * (density0[OPS_ACC5(0,0,0)]) * grad2 * limiter * limiter;
  }
}


#endif
