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

void apply_stencil(const double *A, double *Anew, double *error) {
  Anew[OPS_ACC1(0,0)] = 0.25f * ( A[OPS_ACC0(1,0)] + A[OPS_ACC0(-1,0)]
      + A[OPS_ACC0(0,-1)] + A[OPS_ACC0(0,1)]);
  *error = fmax( *error, fabs(Anew[OPS_ACC1(0,0)]-A[OPS_ACC0(0,0)]));
}
