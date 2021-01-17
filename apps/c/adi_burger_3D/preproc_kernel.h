#ifndef PREPROC_KERNEL_H
#define PREPROC_KERNEL_H

void CopyUVW(const ACC<double> &u, const ACC<double> &v, const ACC<double> &w,
             ACC<double> &uStar, ACC<double> &vStar, ACC<double> &wStar) {
    uStar(0, 0, 0) = u(0, 0, 0);
    vStar(0, 0, 0) = v(0, 0, 0);
    wStar(0, 0, 0) = w(0, 0, 0);
}

void preprocessX(const ACC<double> &u, const ACC<double> &v,
                 const ACC<double> &w, const double *time, ACC<double> &a,
                 ACC<double> &b, ACC<double> &c, ACC<double> &du,
                 ACC<double> &dv, ACC<double> &dw, ACC<double> resU,
                 ACC<double> resV, ACC<double> resW, int *idx) {
    resU(0, 0, 0) = 0;
    resV(0, 0, 0) = 0;
    resW(0, 0, 0) = 0;
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    const double t{*time};
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1 ||
        idx[2] == 0 || idx[2] == nz - 1) {
        a(0, 0, 0) = 0;
        b(0, 0, 0) = 1;
        c(0, 0, 0) = 0;
        du(0, 0, 0) = 2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dv(0, 0, 0) = -2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dw(0, 0, 0) = 2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
    } else {
        a(0, 0, 0) = -r1 * u(0, 0, 0) - r2;
        b(0, 0, 0) = 1 + 2 * r2;
        c(0, 0, 0) = r1 * u(0, 0, 0) - r2;
        du(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * u(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * u(0, 1, 0) +
                      (r1 * w(0, 0, 0) + r2) * u(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * u(0, 0, 1) +
                      (1 - 4 * r2) * u(0, 0, 0);
        dv(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * v(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * v(0, 1, 0) +
                      (r1 * w(0, 0, 0) + r2) * v(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * v(0, 0, 1) +
                      (1 - 4 * r2) * v(0, 0, 0);
        dw(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * w(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * w(0, 1, 0) +
                      (r1 * w(0, 0, 0) + r2) * w(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * w(0, 0, 1) +
                      (1 - 4 * r2) * w(0, 0, 0);
    }
}

void preprocessY(const ACC<double> &u, const ACC<double> &v,
                 const ACC<double> &w, const double *time, ACC<double> &a,
                 ACC<double> &b, ACC<double> &c, ACC<double> &du,
                 ACC<double> &dv, ACC<double> &dw, ACC<double> resU,
                 ACC<double> resV, ACC<double> resW, int *idx) {
    resU(0, 0, 0) = 0;
    resV(0, 0, 0) = 0;
    resW(0, 0, 0) = 0;
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    const double t{*time};
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1 ||
        idx[2] == 0 || idx[2] == nz - 1) {
        a(0, 0, 0) = 0;
        b(0, 0, 0) = 1;
        c(0, 0, 0) = 0;
        du(0, 0, 0) = 2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dv(0, 0, 0) = -2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dw(0, 0, 0) = 2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
    } else {
        a(0, 0, 0) = -r1 * v(0, 0, 0) - r2;
        b(0, 0, 0) = 1 + 2 * r2;
        c(0, 0, 0) = r1 * v(0, 0, 0) - r2;
        du(0, 0, 0) = (r1 * u(0, 0, 0) + r2) * u(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * u(1, 0, 0) +
                      (r1 * w(0, 0, 0) + r2) * u(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * u(0, 0, 1) +
                      (1 - 4 * r2) * u(0, 0, 0);
        dv(0, 0, 0) = (r1 * u(0, 0, 0) + r2) * v(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * v(1, 0, 0) +
                      (r1 * w(0, 0, 0) + r2) * v(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * v(0, 0, 1) +
                      (1 - 4 * r2) * v(0, 0, 0);
        dw(0, 0, 0) = (r1 * u(0, 0, 0) + r2) * w(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * w(1, 0, 0) +
                      (r1 * w(0, 0, 0) + r2) * w(0, 0, -1) +
                      (-r1 * w(0, 0, 0) + r2) * w(0, 0, 1) +
                      (1 - 4 * r2) * w(0, 0, 0);
    }
}

void preprocessZ(const ACC<double> &u, const ACC<double> &v,
                 const ACC<double> &w, const double *time, ACC<double> &a,
                 ACC<double> &b, ACC<double> &c, ACC<double> &du,
                 ACC<double> &dv, ACC<double> &dw, ACC<double> resU,
                 ACC<double> resV, ACC<double> resW, int *idx) {
    resU(0, 0, 0) = 0;
    resV(0, 0, 0) = 0;
    resW(0, 0, 0) = 0;
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    const double t{*time};
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1 ||
        idx[2] == 0 || idx[2] == nz - 1) {
        a(0, 0, 0) = 0;
        b(0, 0, 0) = 1;
        c(0, 0, 0) = 0;
        du(0, 0, 0) = 2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dv(0, 0, 0) = -2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
        dw(0, 0, 0) = 2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
                      (Re * (cos(x) * sin(y) * cos(z) + 1));
    } else {
        a(0, 0, 0) = -r1 * w(0, 0, 0) - r2;
        b(0, 0, 0) = 1 + 2 * r2;
        c(0, 0, 0) = r1 * w(0, 0, 0) - r2;
        du(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * u(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * u(0, 1, 0) +
                      (r1 * u(0, 0, 0) + r2) * u(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * u(1, 0, 0) +
                      (1 - 4 * r2) * u(0, 0, 0);
        dv(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * v(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * v(0, 1, 0) +
                      (r1 * u(0, 0, 0) + r2) * v(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * v(1, 0, 0) +
                      (1 - 4 * r2) * v(0, 0, 0);
        dw(0, 0, 0) = (r1 * v(0, 0, 0) + r2) * w(0, -1, 0) +
                      (-r1 * v(0, 0, 0) + r2) * w(0, 1, 0) +
                      (r1 * u(0, 0, 0) + r2) * w(-1, 0, 0) +
                      (-r1 * u(0, 0, 0) + r2) * w(1, 0, 0) +
                      (1 - 4 * r2) * w(0, 0, 0);
    }
}

#endif  // PREPROC_KERNEL_H
