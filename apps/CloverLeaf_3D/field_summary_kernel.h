#ifndef FIELD_SUMMARY_KERNEL_H
#define FIELD_SUMMARY_KERNEL_H

void field_summary_kernel( const double *volume, const double *density0,
                     const double *energy0, const double *pressure,
                     const double *xvel0,
                     const double *yvel0,
                     const double *zvel0,
                     double *vol,
                     double *mass,
                     double *ie,
                     double *ke,
                     double *press) {

  double vsqrd, cell_vol, cell_mass;
  
  vsqrd = 0.0;
  vsqrd+=0.125*( xvel0[OPS_ACC4(0,0,0)] * xvel0[OPS_ACC4(0,0,0)] +
                 yvel0[OPS_ACC5(0,0,0)] * yvel0[OPS_ACC5(0,0,0)] +
                 zvel0[OPS_ACC6(0,0,0)] * zvel0[OPS_ACC6(0,0,0)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(1,0,0)] * xvel0[OPS_ACC4(1,0,0)] +
                 yvel0[OPS_ACC5(1,0,0)] * yvel0[OPS_ACC5(1,0,0)] +
                 zvel0[OPS_ACC6(1,0,0)] * zvel0[OPS_ACC6(1,0,0)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(0,1,0)] * xvel0[OPS_ACC4(0,1,0)] +
                 yvel0[OPS_ACC5(0,1,0)] * yvel0[OPS_ACC5(0,1,0)] +
                 zvel0[OPS_ACC6(0,1,0)] * zvel0[OPS_ACC6(0,1,0)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(1,1,0)] * xvel0[OPS_ACC4(1,1,0)] +
                 yvel0[OPS_ACC5(1,1,0)] * yvel0[OPS_ACC5(1,1,0)] +
                 zvel0[OPS_ACC6(1,1,0)] * zvel0[OPS_ACC6(1,1,0)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(0,0,1)] * xvel0[OPS_ACC4(0,0,1)] +
                 yvel0[OPS_ACC5(0,0,1)] * yvel0[OPS_ACC5(0,0,1)] +
                 zvel0[OPS_ACC6(0,0,1)] * zvel0[OPS_ACC6(0,0,1)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(1,0,1)] * xvel0[OPS_ACC4(1,0,1)] +
                 yvel0[OPS_ACC5(1,0,1)] * yvel0[OPS_ACC5(1,0,1)] +
                 zvel0[OPS_ACC6(1,0,1)] * zvel0[OPS_ACC6(1,0,1)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(0,1,1)] * xvel0[OPS_ACC4(0,1,1)] +
                 yvel0[OPS_ACC5(0,1,1)] * yvel0[OPS_ACC5(0,1,1)] +
                 zvel0[OPS_ACC6(0,1,1)] * zvel0[OPS_ACC6(0,1,1)]);
  vsqrd+=0.125*( xvel0[OPS_ACC4(1,1,1)] * xvel0[OPS_ACC4(1,1,1)] +
                 yvel0[OPS_ACC5(1,1,1)] * yvel0[OPS_ACC5(1,1,1)] +
                 zvel0[OPS_ACC6(1,1,1)] * zvel0[OPS_ACC6(1,1,1)]);
/* OpenACC doesn't work with nested loops
  for (int iz=0;iz<2;iz++){
    for (int iy=0;iy<2;iy++){
      for (int ix=0;ix<2;ix++){
        vsqrd+=0.125*( xvel0[OPS_ACC4(ix,iy,iz)] * xvel0[OPS_ACC4(ix,iy,iz)] +
                       yvel0[OPS_ACC5(ix,iy,iz)] * yvel0[OPS_ACC5(ix,iy,iz)] +
                       zvel0[OPS_ACC6(ix,iy,iz)] * zvel0[OPS_ACC6(ix,iy,iz)]);
      }
    }
  }
*/
  cell_vol = volume[OPS_ACC0(0,0,0)];
  cell_mass = cell_vol * density0[OPS_ACC1(0,0,0)];
  *vol = *vol + cell_vol;
  *mass = *mass + cell_mass;
  *ie = *ie + cell_mass * energy0[OPS_ACC2(0,0,0)];
  *ke = *ke + cell_mass * 0.5 * vsqrd;
  *press = *press + cell_vol * pressure[OPS_ACC3(0,0,0)];

}

#endif
