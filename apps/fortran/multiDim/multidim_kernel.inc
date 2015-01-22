subroutine multidim_kernel(val, idx)
  IMPLICIT NONE
  REAL(kind=8)   , DIMENSION(1) :: val
  INTEGER(kind=4), DIMENSION(2), INTENT(IN) :: idx

  ! These and the function signature can be code generated
  INTEGER multi_d1
  INTEGER xdim1

  val(OPS_ACC_MD1(0,0,0)) = idx(1)
  val(OPS_ACC_MD1(1,0,0)) = idx(2)
end subroutine