SUBROUTINE set_zero_kernel(A)

    REAL(KIND=8) , dimension(1) :: A
    
    A(OPS_ACC1(0,0)) = 0.0_8

END SUBROUTINE
