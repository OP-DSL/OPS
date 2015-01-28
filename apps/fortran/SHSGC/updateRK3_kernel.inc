subroutine updateRK3_kernel(rho_new, rhou_new, rhoE_new, rho_old, &
                             & rhou_old, rhoE_old, rho_res, rhou_res, rhoE_res, a1, a2)

  real (kind=8) , INTENT(in), DIMENSION(1) :: rho_new, rhou_new, rhoE_new, rho_old, rhou_old, rhoE_old
  real (kind=8), DIMENSION(2), INTENT(IN) :: rho_res, rhou_res, rhoE_res
  real(8) :: a1, a2




end subroutine