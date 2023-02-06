subroutine multidim_reduce_kernel(val, gbl_test, redu_dat1)
  IMPLICIT NONE
  REAL   (kind=8), DIMENSION(2), INTENT(IN) :: val, gbl_test
  REAL(kind=8), DIMENSION(2) :: redu_dat1
  redu_dat1(1) = redu_dat1(1) + val(OPS_ACC_MD1(1,0,0)) * gbl_test(1)
  redu_dat1(2) = redu_dat1(2) + val(OPS_ACC_MD1(2,0,0)) * gbl_test(2)/2.0_8
end subroutine
