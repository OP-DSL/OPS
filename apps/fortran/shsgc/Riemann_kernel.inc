subroutine Riemann_kernel(rho_new, rhou_new, rhoE_new, alam, r, al)

  real (kind=8) , DIMENSION(1) :: rho_new, rhou_new, rhoE_new
  real (kind=8), DIMENSION(3) :: alam
  real (kind=8), DIMENSION(9) :: r
  real (kind=8), DIMENSION(3) :: al


  real(8) :: rl, rr, rho, leftu, rightu, u, hl, hr, h, Vsq, csq, c, g
  real(8) :: dw1, dw2, dw3, delpc2, rdeluc
  real(8) :: fni, p

  integer m

  rl = dsqrt(rho_new(OPS_ACC1(0)))
  rr = dsqrt(rho_new(OPS_ACC1(1)))
  rho = rl + rr
  u = ((rhou_new(OPS_ACC2(0)) / rl) + (rhou_new(OPS_ACC2(1)) / rr)) / rho
  fni = rhou_new(OPS_ACC2(0)) * rhou_new(OPS_ACC2(0)) / rho_new(OPS_ACC1(0))
  p = gam1 * (rhoE_new(OPS_ACC3(0)) - 0.5_8 * fni)
  hl = (rhoE_new(OPS_ACC3(0)) + p)  / rl
  fni = rhou_new(OPS_ACC2(1)) * rhou_new(OPS_ACC2(1)) / rho_new(OPS_ACC1(1))
  p = gam1 * (rhoE_new(OPS_ACC3(1)) - 0.5_8 * fni)
  hr = (rhoE_new(OPS_ACC3(1)) + p)  / rr
  h = (hl + hr)/rho
  Vsq = u*u
  csq = gam1 * (h - 0.5_8 * Vsq)
  g = gam1 / csq
  c = dsqrt(csq)

  alam(OPS_ACC_MD4(1,0)) = u - c
  alam(OPS_ACC_MD4(2,0)) = u
  alam(OPS_ACC_MD4(3,0)) = u + c

  r(OPS_ACC_MD5(1,0)) = 1.0_8
  r(OPS_ACC_MD5(2,0)) = 1.0_8
  r(OPS_ACC_MD5(3,0)) = 1.0_8

  r(OPS_ACC_MD5(4,0)) = u - c
  r(OPS_ACC_MD5(5,0)) = u
  r(OPS_ACC_MD5(6,0)) = u + c

  r(OPS_ACC_MD5(7,0)) = h - u * c
  r(OPS_ACC_MD5(8,0)) = 0.5_8 * Vsq
  r(OPS_ACC_MD5(9,0)) = h + u * c

  DO m = 1,9
    r(OPS_ACC_MD5(m,0)) = r(OPS_ACC_MD5(m,0)) / csq
  END DO

  dw1 = rho_new(OPS_ACC1(1)) - rho_new(OPS_ACC1(0))
  dw2 = rhou_new(OPS_ACC2(1)) - rhou_new(OPS_ACC2(0))
  dw3 = rhoE_new(OPS_ACC3(1)) - rhoE_new(OPS_ACC3(0))

  delpc2 = gam1 * ( dw3 + 0.5_8 * Vsq * dw1  - u * dw2) / csq
  rdeluc = ( dw2 - u * dw1) / c

  al(OPS_ACC_MD6(1,0)) = 0.5_8 * (delpc2 - rdeluc)
  al(OPS_ACC_MD6(2,0)) = dw1 - delpc2
  al(OPS_ACC_MD6(3,0)) = 0.5_8 * ( delpc2 + rdeluc )

  DO m = 1, 3
    al(OPS_ACC_MD6(m,0)) = al(OPS_ACC_MD6(m,0)) * csq
  END DO




end subroutine