#ifndef VISCOSITY_KERNEL_H
#define VISCOSITY_KERNEL_H


void viscosity_kernel( const ACC<double> &xvel0, const ACC<double> &yvel0,
                       const ACC<double> &celldx, const ACC<double> &celldy,
                       const ACC<double> &pressure, const ACC<double> &density0,
                       ACC<double> &viscosity) {

  double ugrad, vgrad,
         grad2,
         pgradx,pgrady,
         pgradx2,pgrady2,
         grad,
         ygrad, xgrad,
         div,
         strain2,
         limiter,
         pgrad;

  //int s2D_00_P10_0P1_P1P1[]  = {0,0, 1,0, 0,1, 1,1};

  ugrad = (xvel0(1,0) + xvel0(1,1)) - (xvel0(0,0) + xvel0(0,1));
  vgrad = (yvel0(0,1) + yvel0(1,1)) - (yvel0(0,0) + yvel0(1,0));

  div = (celldx(0,0))*(ugrad) + (celldy(0,0))*(vgrad);

  strain2 = 0.5*(xvel0(0,1) + xvel0(1,1) - xvel0(0,0) - xvel0(1,0))/(celldy(0,0)) +
            0.5*(yvel0(1,0) + yvel0(1,1) - yvel0(0,0) - yvel0(0,1))/(celldx(0,0));


  //int s2D_10_M10_01_0M1[]  = {1,0, -1,0, 0,1, 0,-1};
  pgradx  = (pressure(1,0) - pressure(-1,0))/(celldx(0,0)+ celldx(1,0));
  pgrady = (pressure(0,1) - pressure(0,-1))/(celldy(0,0)+ celldy(0,1));

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;

  limiter = ((0.5*(ugrad)/celldx(0,0)) * pgradx2 +
             (0.5*(vgrad)/celldy(0,0)) * pgrady2 +
              strain2 * pgradx * pgrady)/ MAX(pgradx2 + pgrady2 , 1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        viscosity(0,0) = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, fabs(pgrady)), pgrady);
    pgrad = sqrt(pgradx*pgradx + pgrady*pgrady);
    xgrad = fabs(celldx(0,0) * pgrad/pgradx);
    ygrad = fabs(celldy(0,0) * pgrad/pgrady);
    grad  = MIN(xgrad,ygrad);
    grad2 = grad*grad;

    viscosity(0,0) = 2.0 * (density0(0,0)) * grad2 * limiter * limiter;
  }
}


#endif
