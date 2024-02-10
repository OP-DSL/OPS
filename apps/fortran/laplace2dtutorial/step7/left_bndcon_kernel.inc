SUBROUTINE left_bndcon_kernel(A, idx)

    REAL(KIND=8), DIMENSION(1) :: A
    INTEGER(KIND=4), DIMENSION(2), INTENT(IN) :: idx

    A(OPS_ACC1(0,0)) = SIN(pi * idx(2) / (jmax+2))

END SUBROUTINE
