MODULE OPS_CONSTANTS
#ifdef OPS_WITH_CUDAFOR
use cudafor
real(8),constant :: dx
real(8),constant :: dy
#else
real(8) :: dx
real(8) :: dy
!$acc declare create(dx,dy)
#endif
END MODULE OPS_CONSTANTS
