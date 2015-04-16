subroutine drhoEpudx_kernel(rhou_new, rho_new, rhoE_new, rhoE_res)

  real (kind=8) , INTENT(in), DIMENSION(1) :: rhou_new, rho_new, rhoE_new
  real (kind=8) , DIMENSION(1) :: rhoE_res
  real (kind=8) :: fni, p, fnim1, fnim2, fnip1, fnip2, deriv

  fni = rhou_new(OPS_ACC1(0)) * rhou_new(OPS_ACC1(0)) / rho_new(OPS_ACC2(0)) ;
  p = gam1 * (rhoE_new(OPS_ACC3(0)) - 0.5 * fni);
  fni = (rhoE_new(OPS_ACC3(0)) + p) * rhou_new(OPS_ACC1(0)) / rho_new(OPS_ACC2(0)) ;

  fnim1 = rhou_new(OPS_ACC1(-1)) * rhou_new(OPS_ACC1(-1)) / rho_new(OPS_ACC2(-1));
  p = gam1 * (rhoE_new(OPS_ACC3(-1)) - 0.5 * fnim1);
  fnim1 = (rhoE_new(OPS_ACC3(-1)) + p) * rhou_new(OPS_ACC1(-1)) / rho_new(OPS_ACC2(-1));

  fnim2 = rhou_new(OPS_ACC1(-2)) * rhou_new(OPS_ACC1(-2)) / rho_new(OPS_ACC2(-2));
  p = gam1 * (rhoE_new(OPS_ACC3(-2)) - 0.5 * fnim2);
  fnim2 = (rhoE_new(OPS_ACC3(-2)) + p ) * rhou_new(OPS_ACC1(-2)) / rho_new(OPS_ACC2(-2));

  fnip1 = rhou_new(OPS_ACC1(1)) * rhou_new(OPS_ACC1(1)) / rho_new(OPS_ACC2(1));
  p = gam1 * (rhoE_new(OPS_ACC3(1)) - 0.5 * fnip1);
  fnip1 = (rhoE_new(OPS_ACC3(1)) + p) * rhou_new(OPS_ACC1(1)) / rho_new(OPS_ACC2(1));

  fnip2 = rhou_new(OPS_ACC1(2)) * rhou_new(OPS_ACC1(2)) / rho_new(OPS_ACC2(2));
  p = gam1 * (rhoE_new(OPS_ACC3(2)) - 0.5 * fnip2);
  fnip2 = (rhoE_new(OPS_ACC3(2)) + p) * rhou_new(OPS_ACC1(2)) / rho_new(OPS_ACC2(2));

  deriv = (fnim2 - fnip2 + 8.0* (fnip1 - fnim1))/(12.00*dx);
  rhoE_res(OPS_ACC4(0)) = deriv;

end subroutine