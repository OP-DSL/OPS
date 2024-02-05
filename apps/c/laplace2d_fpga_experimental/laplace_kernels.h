//Kernels for Laplace demo app
//
void set_zero(ACC<double> &A) {
  A(0,0) = 0.0;
}

void copy(ACC<double> &A, const ACC<double> &Anew) {
  A(0,0) = Anew(0,0);
}

void left_bndcon(ACC<double> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1));
}

void right_bndcon(ACC<double> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1))*exp(-pi);
}

void apply_stencil(const ACC<double> &A, ACC<double> &Anew) {
  Anew(0,0) = 0.25f * ( A(1,0) + A(-1,0)
      + A(0,-1) + A(0,1));
}
