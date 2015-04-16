subroutine vars_kernel(alam, al, gt, cmp, cf)


  real (kind=8), INTENT(in), DIMENSION(3) :: alam, al, gt
  real (kind=8), DIMENSION(3) :: cmp, cf

  real(8) :: anu, aaa, ga, qf, ww
  integer m

  DO m = 1, 3
    anu = alam(OPS_ACC_MD1(m,0));
    aaa = al(OPS_ACC_MD2(m,0));
    ga = aaa * ( gt(OPS_ACC_MD3(m,1)) - gt(OPS_ACC_MD3(m,0))) / (aaa**2.0_8 + del2);
    qf = sqrt ( con + anu**2.0_8);
    cmp(OPS_ACC_MD4(m,0)) = 0.50_8 * qf;
    ww = anu + cmp(OPS_ACC_MD4(m,0)) * ga;
    qf = sqrt(con + ww**2.0_8);
    cf(OPS_ACC_MD5(m,0)) = qf;
  END DO

end subroutine