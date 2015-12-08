subroutine mblock_populate_kernel(val, idx)
  IMPLICIT NONE
  real (kind=8), DIMENSION(1):: val
  INTEGER(kind=4), DIMENSION(1), INTENT(IN) :: idx

  val(OPS_ACC1(0,0)) = (idx(1)-1)+20*(idx(2)-1)
end subroutine