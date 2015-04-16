subroutine update_kernel(rho_new, rhou_new, rhoE_new, s)

  real (kind=8), DIMENSION(1) :: rho_new, rhou_new, rhoE_new
  real (kind=8), INTENT(in), DIMENSION(3) :: s

  rho_new(OPS_ACC1(0))  = rho_new(OPS_ACC1(0))  + s(OPS_ACC_MD4(1,0));
  rhou_new(OPS_ACC2(0)) = rhou_new(OPS_ACC2(0)) + s(OPS_ACC_MD4(2,0));
  rhoE_new(OPS_ACC3(0)) = rhoE_new(OPS_ACC3(0)) + s(OPS_ACC_MD4(3,0));

end subroutine