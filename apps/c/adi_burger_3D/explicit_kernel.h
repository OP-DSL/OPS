#ifndef EXPLICIT_KERNEL_H
#define EXPLICIT_KERNEL_H
void Euler1STCentralDifference(const ACC<double> &uStar, const ACC<double> &vStar,
                               const ACC<double> &wStar, const double *time, ACC<double> u,
                               ACC<double> v, ACC<double> w, int *idx)
{
    const double x{h * idx[0]};
    const double y{h * idx[1]};
    const double z{h * idx[2]};
    const double t{*time};
    if (idx[0] == 0 || idx[0] == nx - 1 || idx[1] == 0 || idx[1] == ny - 1 ||
        idx[2] == 0 || idx[2] == nz - 1)
    {
        //Unsteady problem
        // u(0, 0, 0) = 2 * exp(-t / Re) * sin(x) * sin(y) * cos(z) /
        //               (Re * (cos(x) * sin(y) * cos(z) + 1));
        // v(0, 0, 0) = -2 * exp(-t / Re) * cos(x) * cos(y) * cos(z) /
        //               (Re * (cos(x) * sin(y) * cos(z) + 1));
        // w(0, 0, 0) = 2 * exp(-t / Re) * cos(x) * sin(y) * sin(z) /
        //               (Re * (cos(x) * sin(y) * cos(z) + 1));

        //Steady problem
        u(0, 0, 0) = (-2 * (1 + y + z + y * z)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z));
        v(0, 0, 0) = (-2 * (1 + x + z + x * z)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z));
        w(0, 0, 0) = (-2 * (1 + x + y + x * y)) / (Re * (1 + x + y + x * y + z + x * z + y * z + x * y * z));
    }
    else
    {
        u(0, 0, 0) = uStar(0, 0, 0) - (dt / (2 * h)) * (uStar(0, 0, 0) * (uStar(1, 0, 0) - uStar(-1, 0, 0)) + vStar(0, 0, 0) * (uStar(0, 1, 0) - uStar(0, -1, 0)) + wStar(0, 0, 0) * (uStar(0, 0, 1) - uStar(0, 0, -1))) + (dt / (Re * h * h)) * (uStar(1, 0, 0) + uStar(-1, 0, 0) + uStar(0, 1, 0) + uStar(0, -1, 0) + uStar(0, 0, 1) + uStar(0, 0, -1) - 6 * uStar(0, 0, 0));

        v(0, 0, 0) = vStar(0, 0, 0) - (dt / (2 * h)) * (uStar(0, 0, 0) * (vStar(1, 0, 0) - vStar(-1, 0, 0)) + vStar(0, 0, 0) * (vStar(0, 1, 0) - vStar(0, -1, 0)) + wStar(0, 0, 0) * (vStar(0, 0, 1) - vStar(0, 0, -1))) + (dt / (Re * h * h)) * (vStar(1, 0, 0) + vStar(-1, 0, 0) + vStar(0, 1, 0) + vStar(0, -1, 0) + vStar(0, 0, 1) + vStar(0, 0, -1) - 6 * vStar(0, 0, 0));

        w(0, 0, 0) = wStar(0, 0, 0) - (dt / (2 * h)) * (uStar(0, 0, 0) * (wStar(1, 0, 0) - wStar(-1, 0, 0)) + vStar(0, 0, 0) * (wStar(0, 1, 0) - wStar(0, -1, 0)) + wStar(0, 0, 0) * (wStar(0, 0, 1) - wStar(0, 0, -1))) + (dt / (Re * h * h)) * (wStar(1, 0, 0) + wStar(-1, 0, 0) + wStar(0, 1, 0) + wStar(0, -1, 0) + wStar(0, 0, 1) + wStar(0, 0, -1) - 6 * wStar(0, 0, 0));
    }
}

#endif // EXPLICIT_KERNEL_H