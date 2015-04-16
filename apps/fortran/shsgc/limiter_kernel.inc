subroutine limiter_kernel(al, tht, gt)

  real (kind=8), DIMENSION(3) :: al
  real (kind=8), DIMENSION(3) :: tht
  real (kind=8), DIMENSION(3) :: gt

  real(8) :: aalm, aal, all, ar, gtt
  integer m

  DO m = 1,3
    aalm = abs(al(OPS_ACC_MD1(m,-1)))
    aal = abs(al(OPS_ACC_MD1(m,0)))
    tht(OPS_ACC_MD2(m,0)) = abs (aal - aalm) / (aal + aalm + del2)
    all = al(OPS_ACC_MD1(m,-1))
    ar = al(OPS_ACC_MD1(m,0))
    gtt = all * ( ar * ar + del2 ) + ar * (all * all + del2)
    gt(OPS_ACC_MD3(m,0))= gtt / (ar * ar + all * all + 2.00_8 * del2)
  END DO

end subroutine