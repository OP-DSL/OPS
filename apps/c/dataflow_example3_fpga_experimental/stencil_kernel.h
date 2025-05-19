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

void kernel_1(const ACC<float> &d0, const ACC<float> &d1, ACC<float> &d2, ACC<float> &d3) {
    d2(0,0) = k1 * (d0(-1,0) + d0(1,0)) + k2 * (d1(0,-1) + d1(0,1));
    d3(0,0) = k3 * d0(0,0) + k4 * d1(0,0);
}

void kernel_2(const ACC<float>&d2, const ACC<float> &d3, 
        ACC<float> &d4, ACC<float> &d5) {
    d4(0,0) = d3(0,0) + k5 * d2(0,0);
    d5(0,0) = d2(0,0) + k6 * d3(0,0);
}

void copy(const ACC<float> &in, ACC<float> &out) {
  out(0,0) = in(0,0);
}

#endif //stencil_KERNEL_H
