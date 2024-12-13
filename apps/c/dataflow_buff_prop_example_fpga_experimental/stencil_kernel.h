#ifndef stencil_KERNEL_H
#define stencil_KERNEL_H

void kernel_init_zero(ACC<float> &u) {
    u(0,0) = 0.0;
}

void kernel_idx_init(const int *idx, ACC<float> &u)
{
    u(0,0) = idx[0] + logical_size_x * idx[1];
}

void kernel_const_init(const float* cnst, ACC<float> &u)
{
    u(0,0) = *cnst;
}

// void kernel_const_init_int(const int* cnst, ACC<int> &u)
// {
//     u(0,0) = *cnst;
// }

void kernel_1_5pt(const ACC<float>& a, const ACC<float> &d0, const ACC<float> &d1, 
                            ACC<float> &u1, ACC<float> &u2) {
    u1(0,0) = a(0,0) * 0.2 * (d0(0,0)+d0(-1,0)+d0(1,0)+d0(0,-1)+d0(0,1));
    u2(0,0) = a(0,0) + 0.2 * (d1(0,0)+d1(-1,0)+d1(1,0)+d1(0,-1)+d1(0,1));
    // u1(0,0) = a(0,0) * d0(0,0);
    // u2(0,0) = a(0,0) + d1(0,0);
}

void kernel_2_1pt(const ACC<float>&b, const ACC<float> &d0, 
        const ACC<float> &u1, const ACC<float> &u2, ACC<float>& u3, ACC<float>& u4) {
    u3(0,0) = b(0,0) * u1(0,0) + d0(0,0);
    u4(0,0) = b(0,0) * u2(0,0);
}

void copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //stencil_KERNEL_H
