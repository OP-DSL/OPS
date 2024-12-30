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

void kernel_init_zero_d3(ACC<float> &u) {
    u(0,0,0) = 0.0;
    u(1,0,0) = 0.0;
    u(2,0,0) = 0.0;
}

void kernel_idx_init_d3(const int *idx, ACC<float> &u)
{
    u(0,0,0) = idx[0] + logical_size_x * idx[1];
    u(1,0,0) = idx[0] + logical_size_x * idx[1];
    u(2,0,0) = idx[0] + logical_size_x * idx[1];
}

void kernel_const_init_d3(const float* cnst, ACC<float> &u)
{
    u(0,0,0) = *cnst;
    u(1,0,0) = *cnst;
    u(2,0,0) = *cnst;
}

// void kernel_const_init_int(const int* cnst, ACC<int> &u)
// {
//     u(0,0) = *cnst;
// }

void kernel_1(const ACC<float>& a, const ACC<float> &d0, ACC<float> &d1) {
    d1(0,0,0) = a(0,0) * d0(0,0,0);
    d1(1,0,0) = a(0,0) * d0(1,0,0);
    d1(2,0,0) = a(0,0) * d0(2,0,0);
}

void kernel_2(const ACC<float>&b, const ACC<float> &d0, 
        const ACC<float> &d1, ACC<float> &d2) {
    d2(0,0,0) = b(0,0) * d1(0,0,0) + d0(0,0,0);
    d2(1,0,0) = b(0,0) * d1(1,0,0) + d0(1,0,0);
    d2(2,0,0) = b(0,0) * d1(2,0,0) + d0(2,0,0);
}

void copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

void copy_d3(const ACC<float> &u2, ACC<float> &u) {
  u(0,0,0) = u2(0,0,0);
  u(1,0,0) = u2(1,0,0);
  u(2,0,0) = u2(2,0,0);
}

#endif //stencil_KERNEL_H
