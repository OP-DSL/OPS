#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

void accelerate_stepbymass_kernel( double **density0, double **volume,
                double **stepbymass, double **pressure) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( (*density0[3]) * (*volume[3])
    + (*density0[2]) * (*volume[2])
    + (*density0[0]) * (*volume[0])
    + (*density0[1]) * (*volume[1]) ) * 0.25;

  **stepbymass = 0.5*dt / nodal_mass;

}

void accelerate_kernelx1( double **xvel0, double **xvel1,
                        double **stepbymass,
                        double **xarea, double **pressure) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  **xvel1 = (**xvel0) - (**stepbymass) *
            ( (*xarea[0]) * ( (*pressure[0]) - (*pressure[1]) ) +
              (*xarea[1]) * ( (*pressure[2]) - (*pressure[3]) ) );

}

void accelerate_kernely1( double **yvel0, double **yvel1,
                        double **stepbymass,
                        double **yarea, double **pressure) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};
  **yvel1 = (**yvel0) - (**stepbymass) *
            ( (*yarea[0])  * ( (*pressure[0]) - (*pressure[2]) )  +
              (*yarea[1])  * ( (*pressure[1]) - (*pressure[3]) )  );

}


void accelerate_kernelx2( double **xvel1, double **stepbymass,
                        double **xarea, double **viscosity) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  **xvel1 = (**xvel1) - (**stepbymass) *
            ( (*xarea[0]) * ( (*viscosity[0]) - (*viscosity[1]) ) +
              (*xarea[1]) * ( (*viscosity[2]) - (*viscosity[3]) ) );
}

void accelerate_kernely2( double **yvel1, double **stepbymass,
                        double **yarea, double **viscosity) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  **yvel1 = (**yvel1) - (**stepbymass) *
            ( (*yarea[0]) * ( (*viscosity[0]) - (*viscosity[2]) ) +
              (*yarea[1]) * ( (*viscosity[1]) - (*viscosity[3]) ) );

}
#endif
