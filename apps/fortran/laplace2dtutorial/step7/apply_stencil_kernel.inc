SUBROUTINE apply_stencil_kernel(A, Anew, error)

    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: A
    REAL(KIND=8), DIMENSION(1) :: Anew
    REAL(KIND=8) :: error
    REAL(KIND=8) :: sten_result

    call util_kernel_eqA(A, sten_result)

    Anew(OPS_ACC2(0,0)) = 0.25_8 * sten_result
!    Anew(OPS_ACC2(0,0)) = 0.25_8 * ( A(OPS_ACC1(1,0)) + A(OPS_ACC1(-1,0)) &
!                                 & + A(OPS_ACC1(0,-1)) + A(OPS_ACC1(0,1)) )

    error = MAX( error, ABS( Anew(OPS_ACC2(0,0)) - A(OPS_ACC1(0,0)) ) )

END SUBROUTINE
