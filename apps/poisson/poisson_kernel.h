#ifndef poisson_KERNEL_H
#define poisson_KERNEL_H

void poisson_kernel_populate(const int *dispx, const int *dispy, const int *idx, double *u, double *f, double *ref) {
  double x = dx * (double)(idx[0]+dispx[0]);
  double y = dy * (double)(idx[1]+dispy[0]);
  u[OPS_ACC3(0,0)] = sin(M_PI*x)*cos(2.0*M_PI*y);
  f[OPS_ACC4(0,0)] = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y);
  ref[OPS_ACC5(0,0)] = sin(M_PI*x)*cos(2.0*M_PI*y);

}

void poisson_kernel_initialguess(double *u) {
  u[OPS_ACC0(0,0)] = 0.0;
}

void poisson_kernel_stencil(const double *u, const double *f, double *u2) {
  u2[OPS_ACC2(0,0)] = ((u[OPS_ACC0(-1,0)]+u[OPS_ACC0(1,0)])*dx*dx
                     + (u[OPS_ACC0(0,-1)]+u[OPS_ACC0(0,1)])*dy*dy
                     - dx*dx*dy*dy*f[OPS_ACC1(0,0)])
                     /(2.0*(dx*dx+dy*dy));
}

void poisson_kernel_update(const double *u2, double *u) {
  u[OPS_ACC1(0,0)] = u2[OPS_ACC0(0,0)];
}

void poisson_kernel_error(const double *u, const double *ref, double *err) {
  *err = *err + (u[OPS_ACC0(0,0)]-ref[OPS_ACC1(0,0)])*(u[OPS_ACC0(0,0)]-ref[OPS_ACC1(0,0)]);
}

#endif //poisson_KERNEL_H
