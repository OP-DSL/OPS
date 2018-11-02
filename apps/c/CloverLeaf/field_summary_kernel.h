#ifndef FIELD_SUMMARY_KERNEL_H
#define FIELD_SUMMARY_KERNEL_H

void field_summary_kernel( const ACC<double> &volume, const ACC<double> &density0,
                     const ACC<double> &energy0, const ACC<double> &pressure,
                     const ACC<double> &xvel0,
                     const ACC<double> &yvel0,
                     double *vol,
                     double *mass,
                     double *ie,
                     double *ke,
                     double *press) {

  double vsqrd, cell_vol, cell_mass;

  //xvel0, 0,0, 1,0, 0,1, 1,1
  //yvel0, 0,0, 1,0, 0,1, 1,1

  vsqrd = 0.0;
  vsqrd = vsqrd + 0.25 * ( xvel0(0,0) * xvel0(0,0) + yvel0(0,0) * yvel0(0,0));
  vsqrd = vsqrd + 0.25 * ( xvel0(1,0) * xvel0(1,0) + yvel0(1,0) * yvel0(1,0));
  vsqrd = vsqrd + 0.25 * ( xvel0(0,1) * xvel0(0,1) + yvel0(0,1) * yvel0(0,1));
  vsqrd = vsqrd + 0.25 * ( xvel0(1,1) * xvel0(1,1) + yvel0(1,1) * yvel0(1,1));


  cell_vol = volume(0,0);
  cell_mass = cell_vol * density0(0,0);
  *vol = *vol + cell_vol;
  *mass = *mass + cell_mass;
  *ie = *ie + cell_mass * energy0(0,0);
  *ke = *ke + cell_mass * 0.5 * vsqrd;
  *press = *press + cell_vol * pressure(0,0);

}

#endif
