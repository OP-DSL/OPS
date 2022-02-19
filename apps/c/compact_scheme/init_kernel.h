#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

void initKernel(ACC<double> &u, ACC<double> &ux, ACC<double> &uy,
                ACC<double> &uz, int *idx) {
  const double x{h * idx[0]};
  const double y{h * idx[1]};
  const double z{h * idx[2]};
  u(0, 0, 0) = sin(x)*sin(2*y)*sin(3*z);
  ux(0, 0, 0) = 0;
  uy(0, 0, 0) = 0;
  uz(0, 0, 0) = 0;
}

#endif  // INIT_KERNEL_H