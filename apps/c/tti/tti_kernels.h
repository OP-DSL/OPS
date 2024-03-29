// Description: TTI kernels

void init1(ACC<float> vp, ACC<float> damp) {
    vp(0,0,0) = 1.0F;
    damp(0,0,0) = 1.0F;
}

void tti_kernel1(ACC<float> r11, ACC<float> r12, const ACC<float> u0, const ACC<float> v0, const ACC<float> r3, const ACC<float> r4, const ACC<float> r5) {
    r11(0,0,0) = -(4.16666673e-3F*(u0(0,0,-2) - u0(0,0,2)) + 3.33333338e-2F*(-u0(0,0,-1) + u0(0,0,1)))*r4(0,0,0) - (4.16666673e-3F*(u0(0,-2,0) - u0(0,2,0)) + 3.33333338e-2F*(-u0(0,-1,0) + u0(0,1,0)))*r5(0,0,0) - (4.16666673e-3F*(u0(-2,0,0) - u0(0,0,2)) + 3.33333338e-2F*(-u0(-1,0,0) + u0(1,0,0)))*r3(0,0,0);
    r12(0,0,0) = -(4.16666673e-3F*(v0(0,0,-2) - v0(0,0,2)) + 3.33333338e-2F*(-v0(0,0,-1) + v0(0,0,1)))*r4(0,0,0) - (4.16666673e-3F*(v0(0,-2,0) - v0(0,2,0)) + 3.33333338e-2F*(-v0(0,-1,0) + v0(0,1,0)))*r5(0,0,0) - (4.16666673e-3F*(v0(-2,0,0) - v0(0,0,2)) + 3.33333338e-2F*(-v0(-1,0,0) + v0(1,0,0)))*r3(0,0,0);
}

void tti_kernel2(const ACC<float> r11, const ACC<float> r12, const ACC<float> u0, const ACC<float> v0, const ACC<float> r2, const ACC<float> r3, const ACC<float> r4, const ACC<float> r5, const ACC<float> vp, const ACC<float> damp, const ACC<float> epsilon, const ACC<float> u1, const ACC<float> v1, ACC<float> u2, ACC<float> v2) {

    float r14 = 4.16666673e-3F*(r11(0,0,-2)*r4(0,0,-2) + r11(0,-2,0)*r5(0,-2,0) + r11(-2,0,0)*r3(-2,0,0) - r11(2,0,0)*r3(2,0,0) - r11(0,2,0)*r5(0,2,0) - r11(0,0,2)*r4(0,0,2)) + 3.33333338e-2F*(-r11(0,0,-1)*r4(0,0,-1) - r11(0,-1,0)*r5(0,-1,0) - r11(-1,0,0)*r3(-1,0,0) + r11(1,0,0)*r3(1,0,0) + r11(0,1,0)*r5(0,1,0) + r11(0,0,1)*r4(0,0,1)) + 4.46428561e-6F*(-u0(0,0,-4) - u0(0,-4,0) - u0(-4,0,0) - u0(4,0,0) - u0(0,4,0) - u0(0,0,4)) + 6.34920621e-5F*(u0(0,0,-3) + u0(0,-3,0) + u0(-3,0,0) + u0(3,0,0) + u0(0,3,0) + u0(0,0,3)) + 4.99999989e-4F*(-u0(0,0,-2) - u0(0,-2,0) - u0(-2,0,0) - u0(2,0,0) - u0(0,2,0) - u0(0,0,2)) + 3.99999991e-3F*(u0(0,0,-1) + u0(0,-1,0) + u0(-1,0,0) + u0(1,0,0) + u0(0,1,0) + u0(0,0,1)) - 2.13541662e-2F*u0(0,0,0);
    float r18 = 1.0F/(vp(0,0,0)*vp(0,0,0));
    float r15 = 1.0F/(r10*damp(0,0,0) + r18*r9);
    float r16 = 4.16666673e-3F*(-r12(0,0,-2)*r4(0,0,-2) - r12(0,-2,0)*r5(0,-2,0) - r12(-2,0,0)*r3(-2,0,0) + r12(2,0,0)*r3(2,0,0) + r12(0,2,0)*r5(0,2,0) + r12(0,0,2)*r4(0,0,2));
    float r17 = 3.33333338e-2F*(r12(0,0,-1)*r4(0,0,-1) + r12(0,-1,0)*r5(0,-1,0) + r12(-1,0,0)*r3(-1,0,0) - r12(1,0,0)*r3(1,0,0) - r12(0,1,0)*r5(0,1,0) - r12(0,0,1)*r4(0,0,1));
    u2(0,0,0) = r15*(r10*damp(0,0,0)*u0(0,0,0) + r14*(2*epsilon(0,0,0) + 1) + r18*(-r9*(-2.0F*u0(0,0,0)) - r9*u1(0,0,0)) + (r16 + r17)*r2(0,0,0));
    v2(0,0,0) = r15*(r10*damp(0,0,0)*v0(0,0,0) + r14*r2(0,0,0) + r16 + r17 + r18*(-r9*(-2.0F*v0(0,0,0)) - r9*v1(0,0,0)));
     
}

