subroutine multidim_kernel(val, idx)
  IMPLICIT NONE
  REAL(kind=8)   , DIMENSION(1) :: val
  INTEGER(kind=4), DIMENSION(2), INTENT(IN) :: idx

  val(OPS_ACC_MD0(0,0,0)) = idx(0)
  val(OPS_ACC_MD0(1,0,0)) = idx(1)
end subroutine