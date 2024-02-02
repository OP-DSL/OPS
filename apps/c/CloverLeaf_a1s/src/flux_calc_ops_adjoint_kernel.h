//
//
void flux_calc_kernelx_adjoint(ACC<double> &vol_flux_x,ACC<double> &vol_flux_x_a1s, const ACC<double> &xarea, const ACC<double> &xvel0,ACC<double> &xvel0_a1s, const ACC<double> &xvel1,ACC<double> &xvel1_a1s){
    double temp_a1s;
    double temp_a1s0;
    temp_a1s = dt*0.25*vol_flux_x_a1s(0,0);
    vol_flux_x_a1s(0,0) = 0.0;
    temp_a1s0 = xarea(0,0)*temp_a1s;
    xvel0_a1s(0,0) += + temp_a1s0;
    xvel0_a1s(0,1) += + temp_a1s0;
    xvel1_a1s(0,0) += + temp_a1s0;
    xvel1_a1s(0,1) += + temp_a1s0;
}

void flux_calc_kernely_adjoint(ACC<double> &vol_flux_y,ACC<double> &vol_flux_y_a1s, const ACC<double> &yarea, const ACC<double> &yvel0,ACC<double> &yvel0_a1s, const ACC<double> &yvel1,ACC<double> &yvel1_a1s){
    double temp_a1s;
    double temp_a1s0;
    temp_a1s = dt*0.25*vol_flux_y_a1s(0,0);
    vol_flux_y_a1s(0,0) = 0.0;
    temp_a1s0 = yarea(0,0)*temp_a1s;
    yvel0_a1s(0,0) += + temp_a1s0;
    yvel0_a1s(1,0) += + temp_a1s0;
    yvel1_a1s(0,0) += + temp_a1s0;
    yvel1_a1s(1,0) += + temp_a1s0;
}

