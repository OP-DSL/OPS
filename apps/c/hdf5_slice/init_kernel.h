#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

void initKernelU(ACC<double> &u, int *idx) {
  const double x{h * idx[0]};
  const double y{h * idx[1]};
  const double z{h * idx[2]};
  u(0, 0, 0) = sin(x) * sin(2 * y) * cos(3 * z);
}

void KernelCopy3D(const ACC<double> &u, ACC<float> &u_single) {
#ifdef OPS_3D
  u_single(0, 0, 0) = float(u(0, 0, 0));
#endif
}


void initKernelvelo(ACC<double> &velo, int *idx) {
  const double x{h * idx[0]};
  const double y{h * idx[1]};
  const double z{h * idx[2]};
  velo(0, 0, 0, 0) = sin(x) * sin(y) * sin(z);
  velo(1, 0, 0, 0) = sin(2 * x) * sin(2 * y) * sin(2 * z);
  velo(2, 0, 0, 0) = sin(3 * x) * sin(3 * y) * sin(3 * z);
  ;
}

void initKernelV(ACC<int> &v, int *idx) {
  const double x{h * idx[0]};
  const double y{h * idx[1]};
  const double z{h * idx[2]};
  v(0, 0, 0) = idx[0];
}
#endif // INIT_KERNEL_H