#ifndef stencil_KERNEL_H
#define stencil_KERNEL_H

void kernel_1_5pt(const ACC<float>& a, const ACC<float> &u,
                            ACC<float> &u1) {

    u1(0,0) = a(0,0) * u(0,0);
}

void kernel_2_5pt(const ACC<float>& a, const ACC<float> &u,
                            const ACC<float> &u1, ACC<float> &u2) {

    u2(0,0) = a(0,0) * u1(0,0) + u(0,0);
}

void copy(const ACC<float> &u2, ACC<float> &u) {
  u(0,0) = u2(0,0);
}

#endif //stencil_KERNEL_H
