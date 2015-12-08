subroutine poisson_initialguess_kernel(u)

  real (kind=8), DIMENSION(1) :: u
  u(OPS_ACC1(0,0)) = 0.0_8

end subroutine