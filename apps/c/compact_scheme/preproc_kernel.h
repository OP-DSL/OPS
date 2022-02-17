#ifndef PREPROC_KERNEL_H
#define PREPROC_KERNEL_H
// prepare a, b, c for the linear solver
// at the boundary the compact scheme is downgraded to second-order central
// differencing scheme

void preprocessX(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int i{idx[0]};
  d(0, 0, 0) = u(1, 0, 0) - u(-1, 0, 0);
  if (i == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else if (i == (nx - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}

void preprocessY(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int j{idx[1]};
  d(0, 0, 0) = u(0, 1, 0) - u(0, -1, 0);
  if (j == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;
  } else if (j == (ny - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;
  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}

void preprocessZ(const ACC<double> &u, ACC<double> &a, ACC<double> &b,
                 ACC<double> &c, ACC<double> &d, int *idx) {
  const int k{idx[2]};
  d(0, 0, 0) = u(0, 0, 1) - u(0, 0, -1);
  if (k == 0) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else if (k == (nz - 1)) {
    a(0, 0, 0) = 0;
    b(0, 0, 0) = 2 * h;
    c(0, 0, 0) = 0;

  } else {
    a(0, 0, 0) = left;
    b(0, 0, 0) = present;
    c(0, 0, 0) = right;
  }
}
#endif  // PREPROC_KERNEL_H
