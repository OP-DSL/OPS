subroutine multidim_print_kernel(val)
  IMPLICIT NONE
  REAL   (kind=8), DIMENSION(3) :: val
  !write (*,*) val(OPS_ACC_MD1(1,0,0,0)),val(OPS_ACC_MD1(2,0,0,0)), "const = ", const1
  !print *,val(OPS_ACC_MD1(1,0,0,0)), &
  !val(OPS_ACC_MD1(2,0,0,0)),val(OPS_ACC_MD1(3,0,0,0))
end subroutine
