subroutine multidim_print_kernel(val)
  IMPLICIT NONE
  REAL   (kind=8), DIMENSION(2) :: val
  write (*,*) val(OPS_ACC_MD1(0,0,0)),val(OPS_ACC_MD1(1,0,0)), "const = ", const1
end subroutine