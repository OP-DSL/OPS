#ifndef MULTIDIM_COPY_KERNEL_H
#define MULTIDIM_COPY_KERNEL_H

void multidim_copy_kernel(const ACC<double> &src, ACC<double> &dest){
  dest(0,0,0,0) = src(0,0,0,0);
  dest(1,0,0,0) = src(1,0,0,0);
  dest(2,0,0,0) = src(2,0,0,0);
}

#endif //MULTIDIM_COPY_KERNEL_H
