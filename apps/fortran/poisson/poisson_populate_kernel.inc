subroutine poisson_populate_kernel(dispx, dispy, idx, u, f, ref)
  implicit none
  integer (kind=4), INTENT(IN) :: dispx, dispy
  integer (kind=4), DIMENSION(2), INTENT(IN) :: idx
  real (kind=8), DIMENSION(1) :: u, f, ref

  real(kind=8) :: x, y
  real(kind=8), parameter :: M_PI = 4.D0*ATAN(1.D0) !DATAN could not be used in CUDA FORTRAN

  x = dx * (idx(1)-1.0_8+dispx)
  y = dy * (idx(2)-1.0_8+dispy)

  u(OPS_ACC4(0,0)) = SIN(M_PI*x) * COS(2.0_8*M_PI*y)
  f(OPS_ACC5(0,0)) = -5.0_8 *M_PI*M_PI *SIN(M_PI*x) * COS(2.0_8*M_PI*y)
  ref(OPS_ACC6(0,0)) = SIN(M_PI*x) * COS(2.0_8*M_PI*y)

end subroutine poisson_populate_kernel
