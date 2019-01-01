#ifndef write_kernel_H
#define write_kernel_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_kernel(ACC<double> &mult, ACC<double> &single, ACC<int> &digit, const int *idx) {
  // first argument of multidim is 1
  mult(0, 0, 0, 0) = 1;

  // second argument of multidim is 2
  mult(1, 0, 0, 0) = 2;

  single(0, 0, 0) = 3;

  // format xyz, single integer
  digit(0, 0, 0) = idx[0] * 100 + idx[1] * 10 + idx[2];
}
#endif
