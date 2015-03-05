subroutine poisson_populate_kernel(dispx, dispy, idx, u, f, ref)

  integer (kind=4), DIMENSION(1), INTENT(IN) :: dispx, dispy, idx
  real (kind=8), DIMENSION(1) :: u, f, ref
  real(8) x, y
  real, parameter :: M_PI = 4.D0*DATAN(1.D0)

  x = dx * (idx(1)+dispx(1))
  y = dy * (idx(2)+dispy(1))

  u(OPS_ACC4(0,0)) = sin(M_PI*x)*cos(2.0*M_PI*y)
  f(OPS_ACC5(0,0)) = -5.0*M_PI*M_PI*sin(M_PI*x)*cos(2.0*M_PI*y)
  ref(OPS_ACC6(0,0)) = sin(M_PI*x)*cos(2.0*M_PI*y)

end subroutine