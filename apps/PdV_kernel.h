#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

void PdV_kernel_predict(double **xarea, double **xvel0,
                double **yarea, double **yvel0,
                double **volume_change, double **volume,
                double **pressure,
                double **density0, double **density1,
                double **viscosity,
                double **energy0, double **energy1) {

  double recip_volume, energy_change, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( (*xarea[0]) * ( (*xvel0[0]) + (*xvel0[2]) +
                                (*xvel0[0]) + (*xvel0[2]) ) ) * 0.25 * dt * 0.5;
  right_flux = ( (*xarea[1]) * ( (*xvel0[1]) + (*xvel0[3]) +
                                 (*xvel0[1]) + (*xvel0[3]) ) ) * 0.25 * dt * 0.5;

  bottom_flux = ( (*yarea[0]) * ( (*yvel0[0]) + (*yvel0[1]) +
                                  (*yvel0[0]) + (*yvel0[1]) ) ) * 0.25* dt * 0.5;
  top_flux = ( (*yarea[2]) * ( (*yvel0[2]) + (*yvel0[3]) +
                               (*yvel0[2]) + (*yvel0[3]) ) ) * 0.25 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  **volume_change = (**volume)/(**volume + total_flux);

  min_cell_volume = MIN( (**volume) + right_flux - left_flux + top_flux - bottom_flux ,
                           MIN((**volume) + right_flux - left_flux,
                           (**volume) + top_flux - bottom_flux) );

  recip_volume = 1.0/(**volume);

  energy_change = ( (**pressure)/(**density0) + (**viscosity)/(**density0) ) * total_flux * recip_volume;
  **energy1 = (**energy0) - energy_change;
  **density1 = (**density0) * (**volume_change);

}

void PdV_kernel_nopredict(double **xarea, double **xvel0, double **xvel1,
                double **yarea, double **yvel0, double **yvel1,
                double **volume_change, double **volume,
                double **pressure,
                double **density0, double **density1,
                double **viscosity,
                double **energy0, double **energy1) {


  double recip_volume, energy_change, min_cell_volume;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( (*xarea[0]) * ( (*xvel0[0]) + (*xvel0[2]) +
                                (*xvel1[0]) + (*xvel1[2]) ) ) * 0.25 * dt;
  right_flux = ( (*xarea[1]) * ( (*xvel0[1]) + (*xvel0[3]) +
                                 (*xvel1[1]) + (*xvel1[3]) ) ) * 0.25 * dt;

  bottom_flux = ( (*yarea[0]) * ( (*yvel0[0]) + (*yvel0[1]) +
                                  (*yvel1[0]) + (*yvel1[1]) ) ) * 0.25* dt;
  top_flux = ( (*yarea[2]) * ( (*yvel0[2]) + (*yvel0[3]) +
                               (*yvel1[2]) + (*yvel1[3]) ) ) * 0.25 * dt;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  **volume_change = (**volume)/(**volume + total_flux);

  min_cell_volume = MIN( (**volume) + right_flux - left_flux + top_flux - bottom_flux ,
                           MIN((**volume) + right_flux - left_flux,
                           (**volume) + top_flux - bottom_flux) );

  recip_volume = 1.0/(**volume);

  energy_change = ( (**pressure)/(**density0) + (**viscosity)/(**density0) ) * total_flux * recip_volume;
  **energy1 = (**energy0) - energy_change;
  **density1 = (**density0) * (**volume_change);

}
#endif
