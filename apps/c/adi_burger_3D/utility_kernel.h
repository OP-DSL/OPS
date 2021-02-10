#ifndef UTILITY_KERNEL_H
#define UTILITY_KERNEL_H

void CopyUVW(const ACC<double> &u, const ACC<double> &v, const ACC<double> &w,
             ACC<double> &uStar, ACC<double> &vStar, ACC<double> &wStar) {
    uStar(0, 0, 0) = u(0, 0, 0);
    vStar(0, 0, 0) = v(0, 0, 0);
    wStar(0, 0, 0) = w(0, 0, 0);
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

    // const double uAna{2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
    //                   (Re * (cos(x) * sin(y) * cos(z) + 1))};
    // const double vAna{-2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
    //                   (Re * (cos(x) * sin(y) * cos(z) + 1))};
    // const double wAna{2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
    //                   (Re * (cos(x) * sin(y) * cos(z) + 1))};

    const double uAna{(-2 * (1 + y + z + y * z)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z))};
    const double vAna{(-2 * (1 + x + z + x * z)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z))};
    const double wAna{(-2 * (1 + x + y + x * y)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z))};

    (*uSqr) += (uAna * uAna);
    (*vSqr) += (vAna * vAna);
    (*wSqr) += (wAna * wAna);

    (*uDiffSqr) += ((uAna - u(0, 0, 0)) * (uAna - u(0, 0, 0)));
    (*vDiffSqr) += ((vAna - v(0, 0, 0)) * (vAna - v(0, 0, 0)));
    (*wDiffSqr) += ((wAna - w(0, 0, 0)) * (wAna - w(0, 0, 0)));
}
#endif // UTILITY_KERNEL_H
