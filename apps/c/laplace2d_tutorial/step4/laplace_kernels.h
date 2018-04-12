void set_zero(double *A) {
  A[OPS_ACC0(0,0)] = 0.0;
}

void copy(double *A, const double *Anew) {
  A[OPS_ACC0(0,0)] = Anew[OPS_ACC1(0,0)];
}

void left_bndcon(double *A, const int *idx) {
  A[OPS_ACC0(0,0)] = sin(pi * (idx[1]+1) / (jmax+1));
}

void right_bndcon(double *A, const int *idx) {
  A[OPS_ACC0(0,0)] = sin(pi * (idx[1]+1) / (jmax+1))*exp(-pi);
}
