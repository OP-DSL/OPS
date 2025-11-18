SUBROUTINE apply_stencil_kernel(A, Anew, error)

    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: A
    REAL(KIND=8), DIMENSION(1) :: Anew
    REAL(KIND=8) :: error
    
    REAL(KIND=8) :: Ap10, Am10, A0m1, A0p1

    call helper_kernel_getA(A,Ap10,Am10,A0m1,A0p1)

    Anew(OPS_ACC2(0,0)) = 0.25_8 * ( Ap10 + Am10 &
                                 & + A0m1 + A0p1 )

    error = MAX( error, ABS( Anew(OPS_ACC2(0,0)) - A(OPS_ACC1(0,0)) ) )

END SUBROUTINE
