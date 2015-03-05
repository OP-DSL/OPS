subroutine poisson_populate_kernel(dispx, dispy, idx, u, f, ref)

  integer (kind=4), DIMENSION(1), INTENT(IN) :: dispx, dispy, idx
  real (kind=8), DIMENSION(1) :: u, f, ref
  real(8) x, y
  real(8), parameter :: M_PI = 4.D0*DATAN(1.D0)

  x = dx * (idx(1)-1.0_8+dispx(1))
  y = dy * (idx(2)-1.0_8+dispy(1))
  u(OPS_ACC4(0,0)) = dsin(M_PI*x)*dcos(2.0_8*M_PI*y)
  f(OPS_ACC5(0,0)) = -5.0_8*M_PI*M_PI*dsin(M_PI*x)*dcos(2.0_8*M_PI*y)
  ref(OPS_ACC6(0,0)) = dsin(M_PI*x)*dcos(2.0_8*M_PI*y)

end subroutine