#ifndef RESET_FIELD_KERNEL_H
#define RESET_FIELD_KERNEL_H

inline void reset_field_kernel1( double *density0, double *density1,
                        double *energy0, double *energy1) {

  density0[OPS_ACC0(0,0)]  = density1[OPS_ACC1(0,0)] ;
  energy0[OPS_ACC2(0,0)]  = energy1[OPS_ACC3(0,0)] ;

}

inline void reset_field_kernel2( double *xvel0, double *xvel1,
                        double *yvel0, double *yvel1) {

  xvel0[OPS_ACC0(0,0)]  = xvel1[OPS_ACC1(0,0)] ;
  yvel0[OPS_ACC2(0,0)]  = yvel1[OPS_ACC3(0,0)] ;

}

#endif
