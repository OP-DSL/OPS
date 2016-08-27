#ifndef FIELD_SUMMARY_KERNEL_H
#define FIELD_SUMMARY_KERNEL_H

void field_summary_kernel( const double *volume, const double *density,
                     const double *energy, const double *u,
                     double *vol,
                     double *mass,
                     double *ie,
                     double *temp) {

  double cell_vol, cell_mass;


  cell_vol = volume[OPS_ACC0(0,0)];
  cell_mass = cell_vol * density[OPS_ACC1(0,0)];
  *vol = *vol + cell_vol;
  *mass = *mass + cell_mass;
  *ie = *ie + cell_mass * energy[OPS_ACC2(0,0)];
  *temp = *temp + cell_mass * u[OPS_ACC3(0,0)];
}

#endif
