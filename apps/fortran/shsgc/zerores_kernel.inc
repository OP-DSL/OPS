subroutine zerores_kernel(rho_res, rhou_res, rhoE_res)

  real (kind=8) , DIMENSION(1) :: rho_res, rhou_res, rhoE_res

    rho_res(OPS_ACC1(0))= 0.0_8
    rhou_res(OPS_ACC2(0))= 0.0_8
    rhoE_res(OPS_ACC3(0))= 0.0_8

end subroutine