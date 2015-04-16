subroutine calupwindeff_kernel(cmp, gt, cf, al, ep2, r, eff)


  real (kind=8), INTENT(in), DIMENSION(3) :: cmp, gt, cf, al, r
  real (kind=8), INTENT(in), DIMENSION(9) :: ep2
  real (kind=8), DIMENSION(3) :: eff

  real(8) :: e1, e2, e3

  e1 = (cmp(OPS_ACC_MD1(1,0)) * (gt(OPS_ACC_MD2(1,0)) + gt(OPS_ACC_MD2(1,1))) - cf(OPS_ACC_MD3(1,0)) * al(OPS_ACC_MD4(1,0))) * ep2(OPS_ACC_MD5(1,0))
  e2 = (cmp(OPS_ACC_MD1(2,0)) * (gt(OPS_ACC_MD2(2,0)) + gt(OPS_ACC_MD2(2,1))) - cf(OPS_ACC_MD3(2,0)) * al(OPS_ACC_MD4(2,0))) * ep2(OPS_ACC_MD5(2,0))
  e3 = (cmp(OPS_ACC_MD1(3,0)) * (gt(OPS_ACC_MD2(3,0)) + gt(OPS_ACC_MD2(3,1))) - cf(OPS_ACC_MD3(3,0)) * al(OPS_ACC_MD4(3,0))) * ep2(OPS_ACC_MD5(3,0))

  eff(OPS_ACC_MD7(1,0))=e1 * r(OPS_ACC_MD6(1,0)) + e2 * r(OPS_ACC_MD6(2,0)) + e3 * r(OPS_ACC_MD6(3,0))
  eff(OPS_ACC_MD7(2,0))=e1 * r(OPS_ACC_MD6(4,0)) + e2 * r(OPS_ACC_MD6(5,0)) + e3 * r(OPS_ACC_MD6(6,0))
  eff(OPS_ACC_MD7(3,0))=e1 * r(OPS_ACC_MD6(7,0)) + e2 * r(OPS_ACC_MD6(8,0)) + e3 * r(OPS_ACC_MD6(9,0))

end subroutine