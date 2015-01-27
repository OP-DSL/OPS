subroutine initialize_kernel(x, rho_new, rhou_new, rhoE_new, rhoin, idx)
  real (kind=8) , DIMENSION(1) :: x, rho_new, rhou_new, rhoE_new, rhoin
  INTEGER(kind=4), DIMENSION(1), INTENT(IN) :: idx

  !x[OPS_ACC0(0)] = xmin + (idx[0]-2) * dx;
  !if (x[OPS_ACC0(0)] >= -4.0){
  !  rho_new[OPS_ACC1(0)] = 1.0 + eps * sin(lambda *x[OPS_ACC0(0)]);
  !  rhou_new[OPS_ACC2(0)] = ur * rho_new[OPS_ACC1(0)];
  !  rhoE_new[OPS_ACC3(0)] = (pr / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
  !}
  !else {
  !  rho_new[OPS_ACC1(0)] = rhol;
  !  rhou_new[OPS_ACC2(0)] = ul * rho_new[OPS_ACC1(0)];
  !  rhoE_new[OPS_ACC3(0)] = (pl / gam1) + 0.5 * pow(rhou_new[OPS_ACC2(0)],2)/rho_new[OPS_ACC1(0)];
  !}

  !rhoin[OPS_ACC4(0)] = gam1 * (rhoE_new[OPS_ACC3(0)] - 0.5 * rhou_new[OPS_ACC2(0)] * rhou_new[OPS_ACC2(0)] / rho_new[OPS_ACC1(0)]);

end subroutine