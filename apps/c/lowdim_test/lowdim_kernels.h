#ifndef LOWDIM_KERNEL_H
#define LOWDIM_KERNEL_H

void set_val(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void set_valXY(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void set_valYZ(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void set_valXZ(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void set_valXZ2(ACC<double> &datXZ, const ACC<double> &dat2D)
{
    datXZ(0,0,0) = dat2D(0,0,0);
}

void set_valXY2(ACC<double> &datXY, const ACC<double> &dat2D)
{
    datXY(0,0,0) = dat2D(0,0,0);
}

void set_valYZ2(ACC<double> &datYZ, const ACC<double> &dat2D)
{
    datYZ(0,0,0) = dat2D(0,0,0);
}

void set_valX(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}
void set_valY(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}
void set_valZ(ACC<double> &dat, const double *val)
{
    // make up some values
    dat(0,0,0) = *val;
}

void set3D(ACC<double> &dat,  const int *idx)
{
    dat(0,0,0) = idx[0]*3 + idx[1]*2 + idx[2];
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
  if (dat3D(0,0,0) != 21) {
    printf("Error: dat3D(0,0,0) = %f\n", dat3D(0,0,0));
  }
}

void reduct22D(const ACC<double> &dat3D, ACC<double> &dat2D_xz, ACC<double> &dat2D_xy, ACC<double> &dat2D_yz)
{
    dat2D_xz.combine_inc(0,0,0,dat3D(0,0,0));
    dat2D_xy.combine_inc(0,0,0,dat3D(0,0,0));
    dat2D_yz.combine_inc(0,0,0,dat3D(0,0,0));
}

void reduct21D(ACC<double> &dat3D, ACC<double> &dat1D_x, ACC<double> &dat1D_y, ACC<double> &dat1D_z)
{
    dat1D_x.combine_inc(0,0,0,dat3D(0,0,0));
    dat1D_y.combine_inc(0,0,0,dat3D(0,0,0));
    dat1D_z.combine_inc(0,0,0,dat3D(0,0,0));
}
#endif
