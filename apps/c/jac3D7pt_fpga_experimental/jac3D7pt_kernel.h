#ifndef jac2D9pt_KERNEL_H
#define jac2D9pt_KERNEL_H

void kernel_populate(ACC<float> &u) {

  u(0,0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0,0) = 0.0;
}

void jac3D_kernel_stencil(const ACC<float> &u,
                            ACC<float> &u2) {
	  
    float tmp1 = u(0,0,1) * (0.02f);
    float tmp2 = u(0,1,0) * (0.04f);
    float tmp3 = u(-1,0,0) * (0.05f);
    float tmp4 = u(0,0,0) * (0.79f);
    float tmp5 = u(1,0,0) * (0.06f);
    float tmp6 = u(0,-1,0) * (0.03f);
    float tmp7 = u(0,0,-1) * (0.01f);
    float tmp8 = tmp1 + tmp2;
    float tmp9 = tmp3 + tmp4;
    float tmp10 = tmp5 + tmp6;
    float tmp11 = tmp7 + tmp8;
    float tmp12 = tmp9 + tmp10;
    u2(0,0,0) = tmp11 + tmp12;
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0,0) = u2(0,0,0);
}


#endif //jac2D9pt_KERNEL_H
