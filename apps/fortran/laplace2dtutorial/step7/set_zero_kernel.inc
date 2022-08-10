subroutine set_zero_kernel(A)

    implicit none

    real (8) , dimension(1) :: A
    
    A(OPS_ACC1(0,0)) = 0.0_8

end subroutine
