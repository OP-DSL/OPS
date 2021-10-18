#ifndef poisson_KERNEL_H
#define poisson_KERNEL_H

void poisson_kernel_populate(const int *idx, ACC<double> &u, ACC<double> &f, ACC<double> &ref) {
  double x = dx * (double)(idx[0]);
  double y = dy * (double)(idx[1]);
  double z = dz * (double)(idx[2]);

  u(0,0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);
  f(0,0,0) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y)*sin(2.0*M_PI*z);
  ref(0,0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);

}

void poisson_kernel_initialguess(ACC<double> &u) {
  u(0,0,0) = 0.0;
}

void poisson_kernel_stencil(const ACC<double> &u, ACC<double> &u2) {
  u2(0,0,0) = ((u(-1,0,0)-2.0f*u(0,0,0)+u(1,0,0))*0.125f
             + (u(0,-1,0)-2.0f*u(0,0,0)+u(0,1,0))*0.125f
             + (u(0,0,-1)-2.0f*u(0,0,0)+u(0,0,-1))*0.125f
                     + u(0,0,0));
}

void poisson_kernel_update(const ACC<double> &u2, ACC<double> &u) {
  u(0,0,0) = u2(0,0,0);
}

void poisson_kernel_error(const ACC<double> &u, const ACC<double> &ref, double *err) {
  *err = *err + (u(0,0,0)-ref(0,0,0))*(u(0,0,0)-ref(0,0,0));
}

#endif //poisson_KERNEL_H
