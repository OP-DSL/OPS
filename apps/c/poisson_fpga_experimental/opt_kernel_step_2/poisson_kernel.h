#ifndef poisson_KERNEL_H
#define poisson_KERNEL_H

void poisson_kernel_populate(const int *idx, ACC<float> &u, ACC<float> &f, ACC<float> &ref) {
  float x = dx * (float)(idx[0]);
  float y = dy * (float)(idx[1]);

  u(0,0) = myfun(sin(M_PI*x),cos(2.0*M_PI*y))-1.0;
  f(0,0) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y);
  ref(0,0) = sin(M_PI*x)*cos(2.0*M_PI*y);

}

void poisson_kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void poisson_kernel_stencil(const ACC<float> &u,
                            ACC<float> &u2) {
//  u2(0, 0) = ((u(-1, 0) + u(1, 0)) * dy_2 + (u(0, -1) + u(0, 1)) * dx_2 -
//              f(0, 0) * dx_2_dy_2) /
//             (2.0 * (dx_2_plus_dy_2));

//    float tmp1_t = (u(-1, 0) + u(1, 0));
//    float tmp1 = tmp1_t * dy_2;
//    float tmp2_t = (u(0, -1) + u(0, 1));
//    float tmp2 = tmp2_t * dx_2;
//    float tmp3 = f(0, 0);
//    float tmp4_t = tmp1 + tmp2;
//    float tmp4 = tmp4_t - tmp3;
//    float tmp5 = (2.0 * (dx_2_plus_dy_2));
//    u2(0,0) = tmp4 / tmp5;

	  float tmp1_1 = u(-1,0)+ u(1,0);
      float tmp1_2 = u(0,1) + u(0,-1);
      float tmp1 = tmp1_1 + tmp1_2;
	  float tmp2 = ldexpf(u(0,0),-1); //equalent div 0.5f
	  float tmp3 = ldexpf(tmp1,-3); //equalent div 0.125f
	  u2(0,0) = tmp2 + tmp3;
}

void poisson_kernel_update(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

// void poisson_kernel_error(const ACC<float> &u, const ACC<float> &ref, float *err) {
//   *err = *err + (u(0,0)-ref(0,0))*(u(0,0)-ref(0,0));
// }

#endif //poisson_KERNEL_H
