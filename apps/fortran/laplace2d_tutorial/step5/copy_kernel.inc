subroutine copy_kernel(A, Anew)
    implicit none

    real(8), dimension(1), intent(in) :: Anew
    real(8), dimension(1) :: A
    
    A(OPS_ACC1(0,0)) = Anew(OPS_ACC2(0,0))

end subroutine
