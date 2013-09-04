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

    **vol = 5876895.54965;
    **mass = 5876895.54965;
    **ie = 5876895.54965;
    **ke = 5876895.54965;
    **press = 5876895.54965;
}

#endif
