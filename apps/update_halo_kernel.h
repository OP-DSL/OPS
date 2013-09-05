#ifndef UPDATE_HALO_KERNEL_H
#define UPDATE_HALO_KERNEL_H

void update_halo_kernel( double **density0, double **density1,
                         double **energy0, double **energy1,
                         double **pressure,double **viscosity,
                         double **soundspeed, int** fields) {

  *density0[0] = *density0[1];
  *density1[0] = *density1[1];
  *energy0[0] = *energy0[1];
  *energy1[0] = *energy1[1];
  *pressure[0] = *pressure[1];
  *viscosity[0] = *viscosity[1];
  if((int)*fields[1] == 1) *soundspeed[0] = *soundspeed[1];
}

//void update_halo_kernel( double **density0) {
//  *density0[0] = *density0[1];
//}
#endif
