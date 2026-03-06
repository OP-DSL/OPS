subroutine left_bndcon_kernel(A, idx)

    implicit none

    real(8), dimension(1) :: A
    integer(4), dimension(2), intent(in) :: idx

    A(OPS_ACC1(0,0)) = sin(pi * idx(2) / (jmax+2))

end subroutine
