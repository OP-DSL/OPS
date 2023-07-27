//Kernels for Laplace demo app
//
void set_zero(ACC<half> &A) {
  A(0,0) = 0.0;
}

void copy(ACC<half> &A, const ACC<half> &Anew) {
  A(0,0) = Anew(0,0);
}

void left_bndcon(ACC<half> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1));
}

void right_bndcon(ACC<half> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1))*exp(-pi);
}

void apply_stencil(const ACC<half> &A, ACC<half> &Anew, float *error) {
  Anew(0,0) = (half)0.25f * ( A(1,0) + A(-1,0)
      + A(0,-1) + A(0,1));
  *error = fmax( *error, fabs((float)(Anew(0,0)-A(0,0))));
}
