subroutine left_bndcon_kernel(A, idx)

    implicit none

    real(8), dimension(1) :: A
    integer(4), dimension(2), intent(in) :: idx

    integer, parameter :: jmax=4094
    real(8), parameter :: pi = 2.0_8*asin(1.0_8)
        
    A(OPS_ACC1(0,0)) = sin(pi * idx(2) / (jmax+2))

end subroutine
