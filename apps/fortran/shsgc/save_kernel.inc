subroutine save_kernel(rho_old, rhou_old, rhoE_old, rho_new, rhou_new, rhoE_new)

  real (kind=8) , DIMENSION(1) :: rho_old, rhou_old, rhoE_old
  real (kind=8) , INTENT(IN), DIMENSION(1) :: rho_new, rhou_new, rhoE_new

      rho_old(OPS_ACC1(0))=rho_new(OPS_ACC4(0))
      rhou_old(OPS_ACC2(0))=rhou_new(OPS_ACC5(0))
      rhoE_old(OPS_ACC3(0))=rhoE_new(OPS_ACC6(0))

end subroutine