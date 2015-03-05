subroutine poisson_stencil_kernel(u,f,u2)

  real (kind=8), DIMENSION(1), INTENT(IN) :: u,f
  real (kind=8), DIMENSION(1) :: u2

  u2(OPS_ACC3(0,0)) = ((u(OPS_ACC1(-1,0))+u(OPS_ACC1(1,0)))*dx*dx &
                     & + (u(OPS_ACC1(0,-1))+u(OPS_ACC1(0,1)))*dy*dy &
                     & - dx*dx*dy*dy*f(OPS_ACC2(0,0))) &
                     & /(2.0_8*(dx*dx+dy*dy))

end subroutine