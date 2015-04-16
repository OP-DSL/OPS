subroutine fact_kernel(eff, s)

  real (kind=8), DIMENSION(3) :: s
  real (kind=8), INTENT(in), DIMENSION(3) :: eff

  real(8) :: fact
  integer :: m

  DO m = 1, 3
    fact  = 0.50_8 * dt / dx
    s(OPS_ACC_MD2(m,0)) = -fact * (eff(OPS_ACC_MD1(m,0)) - eff(OPS_ACC_MD1(m,-1)))
  END DO

end subroutine