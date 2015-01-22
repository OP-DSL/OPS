subroutine multidim_reduce_kernel(val, redu_dat1)
  IMPLICIT NONE
  REAL   (kind=8), DIMENSION(2), INTENT(IN) :: val
  REAL(kind=8) :: redu_dat1
  redu_dat1 = redu_dat1 + val(OPS_ACC_MD1(0,0,0)) + val(OPS_ACC_MD1(1,0,0))
end subroutine
