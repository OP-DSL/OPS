#ifndef RESET_FIELD_KERNEL_H
#define RESET_FIELD_KERNEL_H


void reset_field_kernel1( double *density0, const double *density1,
                          double *energy0,  const double *energy1) {

  density0[OPS_ACC0(0,0,0)]  = density1[OPS_ACC1(0,0,0)] ;
  energy0[OPS_ACC2(0,0,0)]  = energy1[OPS_ACC3(0,0,0)] ;

}

void reset_field_kernel2( double *xvel0, const double *xvel1,
                          double *yvel0, const double *yvel1,
                          double *zvel0, const double *zvel1) {

  xvel0[OPS_ACC0(0,0,0)]  = xvel1[OPS_ACC1(0,0,0)] ;
  yvel0[OPS_ACC2(0,0,0)]  = yvel1[OPS_ACC3(0,0,0)] ;
  zvel0[OPS_ACC4(0,0,0)]  = zvel1[OPS_ACC5(0,0,0)] ;
}

#endif
