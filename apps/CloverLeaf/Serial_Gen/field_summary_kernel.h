#ifndef FIELD_SUMMARY_KERNEL_H
#define FIELD_SUMMARY_KERNEL_H

void field_summary_kernel( double **volume, double **density0,
                     double **energy0, double **pressure,
                     double **xvel0,
                     double **yvel0,
                     double **vol,
                     double **mass,
                     double **ie,
                     double **ke,
                     double **press) {

  double vsqrd, cell_vol, cell_mass;

  //xvel0, 0,0, 1,0, 0,1, 1,1
  //yvel0, 0,0, 1,0, 0,1, 1,1

  vsqrd = 0.0;
  //vsqrd = vsqrd + 0.25 * ( (*xvel0[0])*(*xvel0[0]) + (*yvel0[0])*(*yvel0[0]) );
  //vsqrd = vsqrd + 0.25 * ( (*xvel0[1])*(*xvel0[1]) + (*yvel0[1])*(*yvel0[1]) );
  //vsqrd = vsqrd + 0.25 * ( (*xvel0[2])*(*xvel0[2]) + (*yvel0[2])*(*yvel0[2]) );
  //vsqrd = vsqrd + 0.25 * ( (*xvel0[3])*(*xvel0[3]) + (*yvel0[3])*(*yvel0[3]) );

  vsqrd = vsqrd + 0.25 * ( pow((*xvel0[0]),2.0) + pow((*yvel0[0]),2.0) );
  vsqrd = vsqrd + 0.25 * ( pow((*xvel0[1]),2.0) + pow((*yvel0[1]),2.0) );
  vsqrd = vsqrd + 0.25 * ( pow((*xvel0[2]),2.0) + pow((*yvel0[2]),2.0) );
  vsqrd = vsqrd + 0.25 * ( pow((*xvel0[3]),2.0) + pow((*yvel0[3]),2.0) );


  cell_vol = **volume;
  cell_mass = cell_vol * (**density0);
  **vol = **vol + cell_vol;
  **mass = **mass + cell_mass;
  **ie = **ie + cell_mass * (**energy0);
  **ke = **ke + cell_mass * 0.5 * vsqrd;
  **press = **press + cell_vol * (**pressure);

}

#endif
