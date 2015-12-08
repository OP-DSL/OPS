subroutine poisson_update_kernel(u2, u)

  real (kind=8), DIMENSION(1), intent(in):: u2
  real (kind=8),  DIMENSION(1) :: u

  u(OPS_ACC2(0,0)) = u2(OPS_ACC1(0,0))

end subroutine