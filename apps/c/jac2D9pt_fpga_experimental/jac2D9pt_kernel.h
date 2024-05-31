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
	  
    u2(0,0) = u(-1,-1) * (-0.07) + u(-1,0) * (-0.08) + u(-1,1) * (-0.01)
            + u(0,-1) * (-0.06) + u(0,0) * (0.36) + u(0,1) * (-0.02)
            + u(1,-1) * (-0.05) + u(1,0) * (-0.04) + u(1,1) * (-0.03);
}

void kernel_copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}


#endif //jac2D9pt_KERNEL_H
