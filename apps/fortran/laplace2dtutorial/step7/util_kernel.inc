SUBROUTINE util_kernel_eqA(A, sten_result)
    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: A
    REAL(KIND=8), INTENT(INOUT) :: sten_result

    sten_result = A(OPS_ACC1(1,0)) + A(OPS_ACC1(-1,0)) &
                & + A(OPS_ACC1(0,-1)) + A(OPS_ACC1(0,1))

END SUBROUTINE util_kernel_eqA
