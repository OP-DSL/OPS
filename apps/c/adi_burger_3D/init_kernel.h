#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

void initKernel(ACC<double> &u, ACC<double> &v, ACC<double> &w, int *idx) {
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};

    u(0, 0, 0) = 0;
    v(0, 0, 0) = 0;
    w(0, 0, 0) = 0;

    // initialisation for steady problem
    // initial condition for unsteady problem
    // u(0, 0, 0) =
    //     2 * sin(x) * sin(y) * cos(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
    // v(0, 0, 0) =
    //     -2 * cos(x) * cos(y) * cos(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
    // w(0, 0, 0) =
    //     2 * cos(x) * sin(y) * sin(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
}

#endif  // INIT_KERNEL_H