MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
  use cudafor
  real(8), constant :: const1_OPS
  real(8) :: const1
#else
real(8) :: const1
#endif

END MODULE OPS_CONSTANTS