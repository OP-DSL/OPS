#ifndef LOWDIM_KERNEL_H
#define LOWDIM_KERNEL_H

void set_val(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void calc(ACC<double> &dat3D, const ACC<double> &dat2D_xy,  const ACC<double> &dat2D_yz, const ACC<double> &dat2D_xz,
    const ACC<double> &dat1D_x,  const ACC<double> &dat1D_y, const ACC<double> &dat1D_z)
{
  dat3D(0,0,0) = dat2D_xy(0,0,0) +
                           dat2D_yz(0,0,0) +
                           dat2D_xz(0,0,0) +
                           dat1D_x(0,0,0) +
                           dat1D_y(0,0,0) +
                           dat1D_z(0,0,0);
}
#endif
