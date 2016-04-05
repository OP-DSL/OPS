subroutine multidim_kernel(val, idx)
  IMPLICIT NONE
  REAL(kind=8)   , DIMENSION(3) :: val
  INTEGER(kind=4), DIMENSION(3), INTENT(IN) :: idx

  val(OPS_ACC_MD1(1,0,0,0)) = idx(1)
  val(OPS_ACC_MD1(2,0,0,0)) = idx(2)
  val(OPS_ACC_MD1(3,0,0,0)) = idx(3)
end subroutine
