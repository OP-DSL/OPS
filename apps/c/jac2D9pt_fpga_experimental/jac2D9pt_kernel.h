#ifndef jac2D9pt_KERNEL_H
#define jac2D9pt_KERNEL_H

void kernel_populate(ACC<float> &u) {

  u(0,0) = myfun();
}

void kernel_initialguess(ACC<float> &u) {
  u(0,0) = 0.0;
}

void jac2D_kernel_stencil(const ACC<float> &u,
                            ACC<float> &u2) {
	  
    float tmp1 = u(-1,-1) * (-0.07f);
    float tmp2 = u(-1,0) * (-0.08f);
    float tmp3 = u(-1,1) * (-0.01f);
    float tmp4 = u(0,-1) * (-0.06f);
    float tmp5 = u(0,0) * (0.36f);
    float tmp6 = u(0,1) * (-0.02f);
    float tmp7 = u(1,-1) * (-0.05f);
    float tmp8 = u(1,0) * (-0.04f);
    float tmp9 = u(1,1) * (-0.03f);
    float tmp10 = tmp1 + tmp2;
    float tmp11 = tmp3 + tmp4;
    float tmp12 = tmp5 + tmp6;
    float tmp13 = tmp7 + tmp8;
    float tmp14 = tmp11 + tmp10;
    float tmp15 = tmp12 + tmp13;
    float tmp16 = tmp14 + tmp15;
    u2(0,0) = tmp9 + tmp16;
    // u2(0,0) = u(-1,-1) * (-0.07) + u(-1,0) * (-0.08) + u(-1,1) * (-0.01)
    //         + u(0,-1) * (-0.06) + u(0,0) * (0.36) + u(0,1) * (-0.02)
    //         + u(1,-1) * (-0.05) + u(1,0) * (-0.04) + u(1,1) * (-0.03);
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}


#endif //jac2D9pt_KERNEL_H
