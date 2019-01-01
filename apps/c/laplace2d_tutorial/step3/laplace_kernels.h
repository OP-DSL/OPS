//Kernels for Laplace demo app
//
void set_zero(ACC<double> &A) {
  A(0,0) = 0.0;
}

void copy(ACC<double> &A, const ACC<double> &Anew) {
  A(0,0) = Anew(0,0);
}

