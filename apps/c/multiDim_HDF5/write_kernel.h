#ifndef write_kernel_H
#define write_kernel_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_kernel(double *arr, double *arr_single, int *arr_integ,
                  const int *idx) {
  // first argument of multidim is 1
  arr[OPS_ACC_MD0(0, 0, 0, 0)] = 1;

  // second argument of multidim is 2
  arr[OPS_ACC_MD0(1, 0, 0, 0)] = 2;

  arr_single[OPS_ACC1(0, 0, 0)] = 3;

  // format xyz, single integer
  arr_integ[OPS_ACC2(0, 0, 0)] = idx[0] * 100 + idx[1] * 10 + idx[2];
}
#endif
