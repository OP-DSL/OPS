#ifndef write_kernel_H
#define write_kernel_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_kernel(double *arr, double *arr_single, const int *idx) {
  // first argument of multidim is 1
  arr[OPS_ACC_MD0(0, 0, 0, 0)] = 1;

  // second argument of multidim is 2
  arr[OPS_ACC_MD0(1, 0, 0, 0)] = 2;

  arr_single[OPS_ACC1(0, 0, 0)] = 3;
}
#endif
