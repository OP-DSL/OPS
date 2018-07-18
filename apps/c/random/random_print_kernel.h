#ifndef RANDOM_PRINT_KERNEL_H
#define RANDOM_PRINT_KERNEL_H

void random_print_kernel(const double *val){
  printf("(%lf %lf) \n",val[OPS_ACC_MD0(0,0,0)],val[OPS_ACC_MD0(1,0,0)]);
}

#endif //RANDOM_PRINT_KERNEL_H
