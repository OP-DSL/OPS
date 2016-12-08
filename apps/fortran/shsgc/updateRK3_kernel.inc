subroutine updateRK3_kernel(rho_new, rhou_new, rhoE_new, rho_old, &
                             & rhou_old, rhoE_old, rho_res, rhou_res, rhoE_res, a1, a2)

  real (kind=8) , DIMENSION(1) :: rho_new, rhou_new, rhoE_new, rho_old, rhou_old, rhoE_old
  real (kind=8), DIMENSION(1), INTENT(IN) :: rho_res, rhou_res, rhoE_res
  real(8) :: a1, a2

  rho_new(OPS_ACC1(0)) = rho_old(OPS_ACC4(0)) + a1 * (-rho_res(OPS_ACC7(0))) * dt
  rhou_new(OPS_ACC2(0)) = rhou_old(OPS_ACC5(0)) + a1 * (-rhou_res(OPS_ACC8(0))) * dt
  rhoE_new(OPS_ACC3(0)) = rhoE_old(OPS_ACC6(0)) + a1 * (-rhoE_res(OPS_ACC9(0))) * dt

  ! update old state
  rho_old(OPS_ACC4(0)) = rho_old(OPS_ACC4(0)) + a2 * (-rho_res(OPS_ACC7(0))) * dt
  rhou_old(OPS_ACC5(0)) = rhou_old(OPS_ACC5(0)) + a2 * (-rhou_res(OPS_ACC8(0))) * dt
  rhoE_old(OPS_ACC6(0)) = rhoE_old(OPS_ACC6(0)) + a2 * (-rhoE_res(OPS_ACC9(0))) * dt


end subroutine
