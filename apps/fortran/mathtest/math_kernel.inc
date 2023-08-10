subroutine math_kernel_init(d_sin, d_cos, d_log, d_exp, idx)
    implicit none
    real (8) , dimension(1) :: d_sin, d_cos, d_log, d_exp
    integer(4), dimension(3), intent(in) :: idx
    integer(4) :: index
    
    index = idx(1)*4*4 + idx(2)*4 + idx(3)

    d_sin(OPS_ACC1(0,0,0)) = REAL(index)/384.0_8
    d_cos(OPS_ACC2(0,0,0)) = REAL(index)/385.0_8
    d_log(OPS_ACC3(0,0,0)) = REAL(index)/386.0_8
    d_exp(OPS_ACC4(0,0,0)) = REAL(index)/387.0_8

end subroutine

subroutine math_kernel_calc(d_sin, d_cos, d_log, d_exp)
    implicit none
    real (8) , dimension(1) :: d_sin, d_cos, d_log, d_exp

    
    d_sin(OPS_ACC1(0,0,0)) = dsin(d_sin(OPS_ACC1(0,0,0)))
    d_cos(OPS_ACC2(0,0,0)) = dcos(d_cos(OPS_ACC2(0,0,0)))
    d_log(OPS_ACC3(0,0,0)) = dlog(d_log(OPS_ACC3(0,0,0)))
    d_exp(OPS_ACC4(0,0,0)) = dexp(d_exp(OPS_ACC4(0,0,0)))

end subroutine
