subroutine drhoudx_kernel(rhou_new, rho_res)

  real (kind=8) , INTENT(in), DIMENSION(1) :: rhou_new
  real (kind=8) , DIMENSION(1) :: rho_res
  real (kind=8) :: fni, fnim1, fnim2, fnip1, fnip2, deriv

        fni = rhou_new(OPS_ACC1(0))
        fnim1 = rhou_new(OPS_ACC1(-1))
        fnim2 = rhou_new(OPS_ACC1(-2))
        fnip1 = rhou_new(OPS_ACC1(1))
        fnip2 = rhou_new(OPS_ACC1(2))

        deriv = (fnim2 - fnip2 + 8.0_8* (fnip1 - fnim1))/(12.00_8*dx);
        rho_res(OPS_ACC2(0))= deriv;

end subroutine