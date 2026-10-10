// Kernels of the lazy_triggers test: tiny point-wise loops whose results the host reads back
// through every OPS call that may observe data.

// A = global index pattern
void k_init(ACC<double> &A, const int *idx) {
  A(0,0) = idx[0] + 100.0 * idx[1];
}

// B = A + g, g is a read-only global the host changes right after the loop is enqueued
void k_addg(const ACC<double> &A, ACC<double> &B, const double *g) {
  B(0,0) = A(0,0) + g[0];
}

void k_scale(const ACC<double> &B, ACC<double> &C) {
  C(0,0) = 2.0 * B(0,0);
}

void k_sum(const ACC<double> &C, double *s) {
  *s += C(0,0);
}

// D(x,y) = B(x-1,y): reads the halo cell the test fills with ops_halo_transfer
void k_shift(const ACC<double> &B, ACC<double> &D) {
  D(0,0) = B(-1,0);
}

void k_copy(const ACC<double> &C, ACC<double> &X) {
  X(0,0) = C(0,0);
}
