MODULE OPS_CONSTANTS
!----------shsgc Vars/Consts--------------
#ifdef OPS_WITH_CUDAFOR
  !use cudafor
  !real(8), constant :: const1_OPS
  !real(8) :: const1
#else
real(8) :: dx
real(8) :: dy
#endif
END MODULE OPS_CONSTANTS