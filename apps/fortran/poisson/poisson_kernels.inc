subroutine poisson_error_kernel(u, ref, err)

  real (kind=8), DIMENSION(1), intent(in):: u, ref
  real (kind=8) :: err

    err = err + (u(OPS_ACC1(0,0))-ref(OPS_ACC2(0,0)))* &
        & (u(OPS_ACC1(0,0))-ref(OPS_ACC2(0,0)))

end subroutine

subroutine poisson_initialguess_kernel(u)

  real (kind=8), DIMENSION(1) :: u
  u(OPS_ACC1(0,0)) = 0.0_8

end subroutine

subroutine poisson_populate_kernel(dispx, dispy, idx, u, f, ref)
  implicit none
  integer (kind=4), INTENT(IN) :: dispx, dispy
  integer (kind=4), DIMENSION(2), INTENT(IN) :: idx
  real (kind=8), DIMENSION(1) :: u, f, ref
  real(8) x, y
  real(8), parameter :: M_PI = 4.D0*ATAN(1.D0) !DATAN could not be used in CUDA FORTRAN

  x = dx * (idx(1)-1.0_8+dispx)
  y = dy * (idx(2)-1.0_8+dispy)
  u(OPS_ACC4(0,0)) = dsin(M_PI*x)*dcos(2.0_8*M_PI*y)
  f(OPS_ACC5(0,0)) = -5.0_8*M_PI*M_PI*dsin(M_PI*x)*dcos(2.0_8*M_PI*y)
  ref(OPS_ACC6(0,0)) = dsin(M_PI*x)*dcos(2.0_8*M_PI*y)

end subroutine
subroutine poisson_stencil_kernel(u,f,u2)

  real (kind=8), DIMENSION(1), INTENT(IN) :: u,f
  real (kind=8), DIMENSION(1) :: u2

  u2(OPS_ACC3(0,0)) = ((u(OPS_ACC1(-1,0))+u(OPS_ACC1(1,0)))*dx*dx &
                     & + (u(OPS_ACC1(0,-1))+u(OPS_ACC1(0,1)))*dy*dy &
                     & - dx*dx*dy*dy*f(OPS_ACC2(0,0))) &
                     & /(2.0_8*(dx*dx+dy*dy))

end subroutine
subroutine poisson_update_kernel(u2, u)

  real (kind=8), DIMENSION(1), intent(in):: u2
  real (kind=8),  DIMENSION(1) :: u

  u(OPS_ACC2(0,0)) = u2(OPS_ACC1(0,0))

end subroutine
