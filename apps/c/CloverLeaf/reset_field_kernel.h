#ifndef RESET_FIELD_KERNEL_H
#define RESET_FIELD_KERNEL_H


void reset_field_kernel1( ACC<double> &density0, const ACC<double> &density1,
                        ACC<double> &energy0, const ACC<double> &energy1) {

  density0(0,0)  = density1(0,0) ;
  energy0(0,0)  = energy1(0,0) ;

}

void reset_field_kernel2( ACC<double> &xvel0, const ACC<double> &xvel1,
                        ACC<double> &yvel0, const ACC<double> &yvel1) {

  xvel0(0,0)  = xvel1(0,0) ;
  yvel0(0,0)  = yvel1(0,0) ;

}

#endif
