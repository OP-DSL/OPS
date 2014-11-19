#ifndef INITIALIZE_KERNEL_H
#define INITIALIZE_KERNEL_H

#include "vars.h"


void initialize_kernel(double *x, int *idx) {
  x[0] = xmin + (idx[0]-2) * dx;
}

#endif