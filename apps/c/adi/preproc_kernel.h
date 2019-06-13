#ifndef PREPROC_KERNEL_H
#define PREPROC_KERNEL_H

void preproc_kernel(const double *u, double *du,
double *ax, double *bx, double *cx, double *ay, double *by, double *cy,
double *az, double *bz, double *cz, int *idx){

  double a, b, c, d;

  if(idx[0]==0 || idx[0]==nx-1 || idx[1]==0 || idx[1]==ny-1 || idx[2]==0 || idx[2]==nz-1) {
    d = 0.0f; // Dirichlet b.c.'s
    a = 0.0f;
    b = 1.0f;
    c = 0.0f;
  } else {
    d = lambda*( u[OPS_ACC0(-1,0,0)] + u[OPS_ACC0(1,0,0)]
               + u[OPS_ACC0(0,-1,0)] + u[OPS_ACC0(0,1,0)]
               + u[OPS_ACC0(0,0,-1)] + u[OPS_ACC0(0,0,1)]
               - 6.0f*u[OPS_ACC0(0,0,0)]);
    a = -0.5f * lambda;
    b =  1.0f + lambda;
    c = -0.5f * lambda;

  }

  du[OPS_ACC1(0,0,0)] = d;
  ax[OPS_ACC2(0,0,0)] = a;
  bx[OPS_ACC3(0,0,0)] = b;
  cx[OPS_ACC4(0,0,0)] = c;
  ay[OPS_ACC5(0,0,0)] = a;
  by[OPS_ACC6(0,0,0)] = b;
  cy[OPS_ACC7(0,0,0)] = c;
  az[OPS_ACC8(0,0,0)] = a;
  bz[OPS_ACC9(0,0,0)] = b;
  cz[OPS_ACC10(0,0,0)] = c;
}


#endif //PREPROC_KERNEL_H
