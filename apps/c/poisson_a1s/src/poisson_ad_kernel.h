#ifndef poisson_ad_KERNEL_H
#define poisson_ad_KERNEL_H

void poisson_kernel_stencil_adjoint(const ACC<double> &u, ACC<double> &u_a1s,
                                    const ACC<double> &f, ACC<double> &f_a1s,
                                    ACC<double> &u2, ACC<double> &u2_a1s) {
  /* u2(0, 0) = ((u(-1, 0) + u(1, 0)) * dy * dy + */
  /*             (u(0, -1) + u(0, 1)) * dx * dx - */
  /*             f(0, 0) * dx * dx * dy * dy) / */
  /*            (2.0 * dx * dx + dy * dy); */
  double div = (2.0 * (dx * dx + dy * dy));
  u_a1s(-1, 0) += u2_a1s(0, 0) * dy * dy / div;
  u_a1s(1, 0) += u2_a1s(0, 0) * dy * dy / div;
  u_a1s(0, -1) += u2_a1s(0, 0) * dx * dx / div;
  u_a1s(0, 1) += u2_a1s(0, 0) * dx * dx / div;
  f_a1s(0, 0) += u2_a1s(0, 0) * -1 * dx * dx * dy * dy / div;
  u2_a1s(0, 0) = 0;
}

void poisson_kernel_update_adjoint(const ACC<double> &u2, ACC<double> &u2_a1s,
                                   ACC<double> &u, ACC<double> &u_a1s) {
  u2_a1s(0, 0) += u_a1s(0, 0);
  u_a1s(0, 0) = 0;
}

void poisson_kernel_error_adjoint(const ACC<double> &u, ACC<double> &u_a1s,
                                  const ACC<double> &ref, ACC<double> &ref_a1s,
                                  double *err, const double *err_a1s) {
  // *err = *err + (u(0, 0) - ref(0, 0)) * (u(0, 0) - ref(0, 0));
  ref_a1s(0, 0) += -2 * (u(0, 0) - ref(0, 0)) * *err_a1s;
  u_a1s(0, 0) += 2 * (u(0, 0) - ref(0, 0)) * *err_a1s;
}

#endif // poisson_ad_KERNEL_H
