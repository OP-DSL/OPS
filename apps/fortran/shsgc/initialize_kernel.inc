subroutine initialize_kernel(x, rho_new, rhou_new, rhoE_new, rhoin, idx)
  real (kind=8) , DIMENSION(1) :: x, rho_new, rhou_new, rhoE_new, rhoin
  INTEGER(kind=4), DIMENSION(1), INTENT(IN) :: idx

  x(OPS_ACC1(0)) = xmin + (idx(1)-2.0_8-1.0_8) * dx
  if (x(OPS_ACC1(0)) .ge. -4.0_8) then
    rho_new(OPS_ACC2(0)) = 1.0_8 + eps * dsin(lambda * x(OPS_ACC1(0)))
    rhou_new(OPS_ACC3(0)) = ur * rho_new(OPS_ACC2(0))
    rhoE_new(OPS_ACC4(0)) = (pr / gam1) + 0.5_8 * (rhou_new(OPS_ACC3(0))**2_8)/rho_new(OPS_ACC2(0))
  else
    rho_new(OPS_ACC2(0)) = rhol
    rhou_new(OPS_ACC3(0)) = ul * rho_new(OPS_ACC2(0))
    rhoE_new(OPS_ACC4(0)) = (pl / gam1) + 0.5_8 * (rhou_new(OPS_ACC3(0))**2_8)/rho_new(OPS_ACC2(0))
  end if

  rhoin(OPS_ACC5(0)) = gam1 * (rhoE_new(OPS_ACC4(0)) - 0.5_8 * rhou_new(OPS_ACC3(0)) * rhou_new(OPS_ACC3(0)) / rho_new(OPS_ACC2(0)))

end subroutine