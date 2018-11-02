#ifndef MULTIDIM_PRINT_KERNEL_H
#define MULTIDIM_PRINT_KERNEL_H

void multidim_print_kernel(const ACC<double> &val){
  printf("(%lf %lf %lf) \n",val(0,0,0,0),val(1,0,0,0),val(2,0,0,0));
}

#endif //MULTIDIM_PRINT_KERNEL_H
