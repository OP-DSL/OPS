//Kernels for Laplace demo app
//
void set_zero(ACC<float> &A) {
  A(0,0) = 0.0f;
}

void copy(ACC<float> &A, const ACC<float> &Anew) {
  A(0,0) = Anew(0,0);
}

void left_bndcon(ACC<float> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1));
}

void right_bndcon(ACC<float> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1))*exp(-pi);
}

void apply_stencil(const ACC<float> &A, ACC<float> &Anew, float *error) {
  Anew(0,0) = 0.25f * ( A(1,0) + A(-1,0)
      + A(0,-1) + A(0,1));
  *error = fmax( *error, fabs(Anew(0,0)-A(0,0)));
}
