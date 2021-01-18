#ifndef INIT_KERNEL_H
#define INIT_KERNEL_H

void initKernel(ACC<double> &u, ACC<double> &v, ACC<double> &w, int *idx) {
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    u(0, 0, 0) =
        2 * sin(x) * sin(y) * cos(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
    v(0, 0, 0) =
        -2 * cos(x) * cos(y) * cos(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
    w(0, 0, 0) =
        2 * cos(x) * sin(y) * sin(z) / (Re * (cos(x) * sin(y) * cos(z) + 1));
}

void calculateL2NormError(const ACC<double> &u, const ACC<double> &v,
                          const ACC<double> &w, const double *time,
                          double *uSqr, double *vSqr, double *wSqr,
                          double *uDiffSqr, double *vDiffSqr, double *wDiffSqr,
                          int *idx) {
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    const double t{*time};

    const double uAna{2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1))};
    const double vAna{-2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1))};
    const double wAna{2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1))};

    (*uSqr) += (uAna * uAna);
    (*vSqr) += (vAna * vAna);
    (*wSqr) += (wAna * wAna);

    (*uDiffSqr) += ((uAna - u(0, 0, 0)) * (uAna - u(0, 0, 0)));
    (*vDiffSqr) += ((vAna - v(0, 0, 0)) * (vAna - v(0, 0, 0)));
    (*wDiffSqr) += ((wAna - w(0, 0, 0)) * (wAna - w(0, 0, 0)));
}

#endif  // INIT_KERNEL_H