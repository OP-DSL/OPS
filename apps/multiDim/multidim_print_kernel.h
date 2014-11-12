#ifndef MULTIDIM_PRINT_KERNEL_H
#define MULTIDIM_PRINT_KERNEL_H

void multidim_print_kernel(double *val) {
  printf("(%lf %lf) \n",val[OPS_ACC_MD0(0,0,0)],val[OPS_ACC_MD0(1,0,0)]);
}

#endif //MULTIDIM_PRINT_KERNEL_H
