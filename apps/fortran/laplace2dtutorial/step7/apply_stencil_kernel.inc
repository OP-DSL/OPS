subroutine apply_stencil_kernel(A, Anew, error)

    implicit none

    real(8), dimension(1), intent(in) :: A
    real(8), dimension(1) :: Anew
    real(8) :: error

    Anew(OPS_ACC2(0,0)) = 0.25_8 * ( A(OPS_ACC1(1,0)) + A(OPS_ACC1(-1,0)) &
                                 & + A(OPS_ACC1(0,-1)) + A(OPS_ACC1(0,1)) )

    error = max( error, abs( Anew(OPS_ACC2(0,0)) - A(OPS_ACC1(0,0)) ) )

end subroutine
