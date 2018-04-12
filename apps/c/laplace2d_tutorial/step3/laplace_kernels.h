void set_zero(double *A) {
  A[OPS_ACC0(0,0)] = 0.0;
}

void copy(double *A, const double *Anew) {
  A[OPS_ACC0(0,0)] = Anew[OPS_ACC1(0,0)];
}
