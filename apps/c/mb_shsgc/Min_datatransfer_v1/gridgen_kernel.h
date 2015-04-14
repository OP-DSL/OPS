#ifndef gridgen_kernel_H
#define gridgen_kernel_H

void gridgen_kernel(double *x, const int *id) {

  x[OPS_ACC0(0)] = xt +  id[0] *dx;

}

#endif