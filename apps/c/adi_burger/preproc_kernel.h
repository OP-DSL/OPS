#ifndef PREPROC_KERNEL_H
#define PREPROC_KERNEL_H

void preprocessX(const ACC<double> &u, const ACC<double> &v, const double *time,
                 ACC<double> &a, ACC<double> &b, ACC<double> &c,
                 ACC<double> &du, ACC<double> &dv, ACC<double> resU,
                 ACC<double> resV, int *idx) {
    resU(0, 0) = 0;
    resV(0, 0) = 0;
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1) {
        a(0, 0) = 0;
        b(0, 0) = 1;
        c(0, 0) = 0;
        du(0, 0) =
            0.75 -
            1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h - (*time)) *
                              Re / 32)));
        dv(0, 0) =
            0.75 +
            1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h - (*time)) *
                              Re / 32)));
    } else {
        a(0, 0) = -r1 * u(0, 0) - r2;
        b(0, 0) = 1 + 2 * r2;
        c(0, 0) = -r1 * u(0, 0) + r2;
        du(0, 0) = (r1 * v(0, 0) + r2) * u(0, -1) + (1 - 2 * r2) * u(0, 0) +
                   (-r1 * v(0, 0) + r2) * u(0, 1);
        dv(0, 0) = (r1 * v(0, 0) + r2) * v(0, -1) + (1 - 2 * r2) * v(0, 0) +
                   (-r1 * v(0, 0) + r2) * v(0, 1);
    }
}

void preprocessY(const ACC<double> &u, const ACC<double> &v, const double *time,
                 ACC<double> &a, ACC<double> &b, ACC<double> &c,
                 ACC<double> &du, ACC<double> &dv, ACC<double> resU,
                 ACC<double> resV, int *idx) {
    resU(0, 0) = 0;
    resV(0, 0) = 0;
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1) {
        a(0, 0) = 0;
        b(0, 0) = 1;
        c(0, 0) = 0;
        du(0, 0) =
            0.75 -
            1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h - (*time)) *
                              Re / 32)));
        dv(0, 0) =
            0.75 +
            1 / (4 * (1 + exp((-4 * idx[0] * h + 4 * idx[1] * h - (*time)) *
                              Re / 32)));
    } else {
        a(0, 0) = -r1 * v(0, 0) - r2;
        b(0, 0) = 1 + 2 * r2;
        c(0, 0) = -r1 * v(0, 0) + r2;
        du(0, 0) = (r1 * u(0, 0) + r2) * u(-1, 0) + (1 - 2 * r2) * u(0, 0) +
                   (-r1 * u(0, 0) + r2) * u(1, 0);
        dv(0, 0) = (r1 * u(0, 0) + r2) * v(-1, 0) + (1 - 2 * r2) * v(0, 0) +
                   (-r1 * u(0, 0) + r2) * v(1, 0);
    }
}
#endif  // PREPROC_KERNEL_H
