SUBROUTINE helper_kernel_getA(A, Ap10, Am10, A0m1, A0p1)

    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: A
    REAL(KIND=8), intent(out) :: Ap10, Am10, A0m1, A0p1

    Ap10 = A(OPS_ACC1(1,0))
    Am10 = A(OPS_ACC1(-1,0))
    A0m1 = A(OPS_ACC1(0,-1))
    A0p1 = A(OPS_ACC1(0,1))

END SUBROUTINE helper_kernel_getA
