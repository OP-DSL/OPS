#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H
void initKernel(ACC<double> &uVel, ACC<double> &vVel, int *idx) {
    uVel(0, 0) =
        0.75 -
        1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h) * Re / 32)));
    vVel(0, 0) =
        0.75 +
        1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h) * Re / 32)));
}
#endif  // INIT_KERNEL_H