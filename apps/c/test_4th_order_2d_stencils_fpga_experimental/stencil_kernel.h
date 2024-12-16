#ifndef stencil_KERNEL_H
#define stencil_KERNEL_H
/*
* The main stencil will be 4th order 2D stencil to functionally verify the OPS HLS
* to support RTM application. This kernels are for functional verification.
*/
void kernel_init_zero(ACC<float>& u) {
    u(0,0) = 0.0;
}

void kernel_idx_init(const int *idx, ACC<float> &u)
{
    u(0,0) = idx[0] + actual_size_x * idx[1];
}

void kernel_const_init(const float* cnst, ACC<float> &u)
{
    u(0,0) = *cnst;
}

void kernel_2D_9pt(const ACC<float>& a, const ACC<float> &d0, ACC<float> &d1) {
    float t0 = d0(-2,0) + d0(-1,0);
    float t1 = d0(1,0) + d0(2,0);
    float t2 = d0(0,-2) + d0(0,-1);
    float t3 = d0(0,1) + d0(0,2);
    float t4 = t0 + t1;
    float t5 = t2 + t3;
    float t6 = t4 + t5;
    d1(0,0) =  a(0,0) * ldexpf(t6,-3);
}

void kernel_2D_5pt(const ACC<float>& a, const ACC<float> &d0, ACC<float> &d1) {
    d1(0,0) = a(0,0) * 0.2 * (d0(0,0)+d0(-1,0)+d0(1,0)+d0(0,-1)+d0(0,1));
}

void copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //stencil_KERNEL_H
