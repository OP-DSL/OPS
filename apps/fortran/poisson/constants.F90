MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
use cudafor
real(8),constant :: dx_opsconstant
real(8),constant :: dy_opsconstant
real(8) :: dx
real(8) :: dy
#else
real(8) :: dx
real(8) :: dy
#endif

END MODULE OPS_CONSTANTS
