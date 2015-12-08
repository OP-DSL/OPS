subroutine multidim_copy_kernel(val1, val2)
  IMPLICIT NONE
  REAL   (kind=8), DIMENSION(2) :: val1
  REAL   (kind=8), DIMENSION(2) :: val2
  val2(OPS_ACC_MD2(1,0,0)) = val1(OPS_ACC_MD1(1,0,0))
  val2(OPS_ACC_MD2(2,0,0)) = val1(OPS_ACC_MD1(2,0,0))
end subroutine