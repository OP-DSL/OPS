#ifndef MULTIDIM_COPY_KERNEL_H
#define MULTIDIM_COPY_KERNEL_H

void multidim_copy_kernel(double *src, double *dest) {
  dest[OPS_ACC_MD1(0,0,0)] = src[OPS_ACC_MD0(0,0,0)];
  dest[OPS_ACC_MD1(1,0,0)] = src[OPS_ACC_MD0(1,0,0)];
}

#endif //MULTIDIM_COPY_KERNEL_H
