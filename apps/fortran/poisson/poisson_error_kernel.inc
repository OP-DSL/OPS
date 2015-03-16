subroutine poisson_error_kernel(u, ref, err)

  real (kind=8), DIMENSION(1), intent(in):: u, ref
  real (kind=8) :: err

    err = err + (u(OPS_ACC1(0,0))-ref(OPS_ACC2(0,0)))* &
        & (u(OPS_ACC1(0,0))-ref(OPS_ACC2(0,0)))

end subroutine
