#ifndef stencil_KERNEL_H
#define stencil_KERNEL_H
/*
* The main stencil will be 8th order stencil to functionally verify the OPS HLS
* to support RTM application. This kernels are for functional verification.
*/
void kernel_init_zero(ACC<float> &u) {
    u(0,0,0) = 0.0;
}

void kernel_idx_init(const int *idx, ACC<float> &u)
{
    u(0,0,0) = idx[0] + logical_size_x * idx[1] + logical_size_x * logical_size_y * idx[2];
}

void kernel_const_init(const float* cnst, ACC<float> &u)
{
    u(0,0,0) = *cnst;
}

void kernel_1_25pt(const ACC<float>& a, const ACC<float> &d0, const ACC<float> &d1, 
                            ACC<float> &u1, ACC<float> &u2) {
    float t0 = d0(-4,0,0) + d0(-3,0,0);
    float t1 = d0(-2,0,0) + d0(-1,0,0);
    float t2 = d0(1,0,0) + d0(2,0,0);
    float t3 = d0(3,0,0) + d0(4,0,0);
    float t4 = d0(0,-4,0) + d0(0,-3,0);
    float t5 = d0(0,-2,0) + d0(0,-1,0);
    float t6 = d0(0,1,0) + d0(0,2,0);
    float t7 = d0(0,4,0) + d0(0,3,0);
    float t8 = d0(0,0,-4) + d0(0,0,-3);
    float t9 = d0(0,0,-2) + d0(0,0,-1);
    float t10 = d0(0,0,1) + d0(0,0,2);
    float t11 = d0(0,0,3) + d0(0,0,4);

    float t12 = t0 + t1;
    float t13 = t2 + t3;
    float t14 = t4 + t5;
    float t15 = t6 + t7;
    float t16 = t8 + t9;
    float t17 = t10 + t11;

    float t18 = t12 + t13;
    float t19 = t14 + t15;
    float t20 = t16 + t17;
    
    float t21 = t18 + t19;
    float t22 = 0.041666666666667f * (t20 + t21);
    u1(0,0,0) = a(0,0,0) * t22;

    float t30 = d1(-4,0,0) + d1(-3,0,0);
    float t31 = d1(-2,0,0) + d1(-1,0,0);
    float t32 = d1(1,0,0) + d1(2,0,0);
    float t33 = d1(3,0,0) + d1(4,0,0);
    float t34 = d1(0,-4,0) + d1(0,-3,0);
    float t35 = d1(0,-2,0) + d1(0,-1,0);
    float t36 = d1(0,1,0) + d1(0,2,0);
    float t37 = d1(0,4,0) + d1(0,3,0);
    float t38 = d1(0,0,-4) + d1(0,0,-3);
    float t39 = d1(0,0,-2) + d1(0,0,-1);
    float t310 = d1(0,0,1) + d1(0,0,2);
    float t311 = d1(0,0,3) + d1(0,0,4);

    float t312 = t30 + t31;
    float t313 = t32 + t33;
    float t314 = t34 + t35;
    float t315 = t36 + t37;
    float t316 = t38 + t39;
    float t317 = t310 + t311;

    float t318 = t312 + t313;
    float t319 = t314 + t315;
    float t320 = t316 + t317;
    
    float t321 = t318 + t319;
    float t322 = 0.041666666666667f * (t320 + t321);
    u2(0,0,0) = a(0,0,0) + t322;
}

void kernel_2_25pt(const ACC<float>&b, const ACC<float> &d0, 
        const ACC<float> &u1, const ACC<float> &u2, ACC<float>& u3, ACC<float>& u4) {
    u3(0,0,0) = b(0,0,0) * u1(0,0,0) + d0(0,0,0);
    u4(0,0,0) = b(0,0,0) * u2(0,0,0);
}

void copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0,0) = u2(0,0,0);
}

#endif //stencil_KERNEL_H
