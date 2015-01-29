subroutine drhouupdx_kernel(rhou_new, rho_new, rhoE_new, rhou_res)

  real (kind=8) , INTENT(in), DIMENSION(1) :: rhou_new, rho_new, rhoE_new
  real (kind=8) , DIMENSION(1) :: rhou_res
  real (kind=8) :: fni, p, fnim1, fnim2, fnip1, fnip2, deriv

      fni = rhou_new(OPS_ACC1(0)) * rhou_new(OPS_ACC1(0)) / rho_new(OPS_ACC2(0))
      p = gam1 * (rhoE_new(OPS_ACC3(0)) - 0.5_8 * fni)
      fni = fni + p
      fnim1 = rhou_new(OPS_ACC1(-1)) * rhou_new(OPS_ACC1(-1)) / rho_new(OPS_ACC2(-1))
      p = gam1 * (rhoE_new(OPS_ACC3(-1)) - 0.5_8 * fnim1)
      fnim1 = fnim1 + p
      fnim2 = rhou_new(OPS_ACC1(-2)) * rhou_new(OPS_ACC1(-2)) / rho_new(OPS_ACC2(-2))
      p = gam1 * (rhoE_new(OPS_ACC3(-2)) - 0.5_8 * fnim2)
      fnim2 = fnim2 + p
      fnip1 = rhou_new(OPS_ACC1(1)) * rhou_new(OPS_ACC1(1)) / rho_new(OPS_ACC2(1))
      p = gam1 * (rhoE_new(OPS_ACC3(1)) - 0.5_8 * fnip1)
      fnip1 = fnip1 + p
      fnip2 = rhou_new(OPS_ACC1(2)) * rhou_new(OPS_ACC1(2)) / rho_new(OPS_ACC2(2))

      p = gam1 * (rhoE_new(OPS_ACC3(2)) - 0.5_8 * fnip2)
      fnip2 = fnip2 + p
      deriv = (fnim2 - fnip2 + 8.0_8* (fnip1 - fnim1))/(12.00_8*dx)
      rhou_res(OPS_ACC4(0)) = deriv

end subroutine