#ifndef VISCOSITY_KERNEL_H
#define VISCOSITY_KERNEL_H

void viscosity_kernel( double **xvel0, double **yvel0,
                       double **celldx, double **celldy,
                       double **pressure, double **density0,
                       double **viscosity) {

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

  ugrad = (*xvel0[1] + *xvel0[3]) - (*xvel0[0] + *xvel0[2]);
  vgrad = (*yvel0[2] + *yvel0[3]) - (*yvel0[0] + *yvel0[1]);

  div = (*celldx[0])*(ugrad)+  (*celldy[0])*(vgrad);

  strain2 = 0.5*(*xvel0[2] + *xvel0[3] - *xvel0[0] - *xvel0[1])/(*celldy[0]) +
            0.5*(*yvel0[1] + *yvel0[3] - *yvel0[0] - *yvel0[2])/(*celldx[0]);


  //int self2D_4point1xy[]  = {1,0, -1,0, 0,1, 0,-1};
  pgradx  = (*pressure[0] - *pressure[1])/(*celldx[0]+ *celldx[1]);
  pgrady = (*pressure[2] - *pressure[3])/(*celldy[0]+ *celldy[1]);

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;

  limiter = ((0.5*(ugrad)/(*celldx[0])) * pgradx2 +
             (0.5*(vgrad)/(*celldy[0])) * pgrady2 +
              strain2 * pgradx * pgrady)/ MAX(pgradx2 + pgrady2 , 1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        **viscosity = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, fabs(pgrady)), pgrady);
    pgrad = sqrt(pgradx*pgradx + pgrady*pgrady);
    xgrad = fabs(*celldx[0] * pgrad/pgradx);
    ygrad = fabs(*celldy[0] * pgrad/pgrady);
    grad  = MIN(xgrad,ygrad);
    grad2 = grad*grad;

    **viscosity = 2.0 * (**density0) * grad2 * limiter * limiter;
  }
}

#endif
