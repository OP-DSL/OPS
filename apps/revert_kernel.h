#ifndef PdV_KERNEL_H
#define PdV_KERNEL_H

void revert_kernel( double **density0, double **density1,
                double **energy0, double **energy1) {

  **density1 = **density0;
  **energy1 = **energy0;
}
#endif
