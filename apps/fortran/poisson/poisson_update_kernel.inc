subroutine poisson_update_kernel(inp, out)

  real (kind=8), DIMENSION(1), intent(in):: inp
  real (kind=8),  DIMENSION(1) :: out

  out(OPS_ACC2(0,0)) = inp(OPS_ACC1(0,0))

end subroutine
