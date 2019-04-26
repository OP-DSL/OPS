#ifndef LOWDIM_KERNEL_H
#define LOWDIM_KERNEL_H

void set_val(double *dat, const double *val)
{
    // make up some values
    dat[OPS_ACC0(0,0,0)] = *val;
}

void calc(double *dat3D, const double *dat2D_xy,  const double *dat2D_yz, const double *dat2D_xz,
    const double *dat1D_x,  const double *dat1D_y, const double *dat1D_z)
{
  dat3D[OPS_ACC0(0,0,0)] = dat2D_xy[OPS_ACC1(0,0,0)] +
                           dat2D_yz[OPS_ACC2(0,0,0)] +
                           dat2D_xz[OPS_ACC3(0,0,0)] +
                           dat1D_x[OPS_ACC4(0,0,0)] +
                           dat1D_y[OPS_ACC5(0,0,0)] +
                           dat1D_z[OPS_ACC6(0,0,0)];
}
#endif
