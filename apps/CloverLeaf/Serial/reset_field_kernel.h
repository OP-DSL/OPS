#ifndef RESET_FIELD_KERNEL_H
#define RESET_FIELD_KERNEL_H

void reset_field_kernel1( double **density0, double **density1,
                        double **energy0, double **energy1) {

  **density0 = **density1;
  **energy0 = **energy1;

}

void reset_field_kernel2( double **xvel0, double **xvel1,
                        double **yvel0, double **yvel1) {

  **xvel0 = **xvel1;
  **yvel0 = **yvel1;

}

#endif
