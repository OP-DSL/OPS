#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

void initKernel(ACC<double> &u, int *idx) {
  const double x{h * idx[0]};
  const double y{h * idx[1]};
  const double z{h * idx[2]};
  u(0, 0, 0) = idx[0];  // sin(x) * sin(2 * y) * cos(3 * z);
  printf("Idx=%d\n", idx[0]);
}
#endif  // INIT_KERNEL_H