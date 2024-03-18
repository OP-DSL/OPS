MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
use cudafor
real(8),constant :: dx_opsconstant
real(8),constant :: dy_opsconstant
real(8) :: dx
real(8) :: dy = 0.01_8
#else
real(8) :: dx
real(8) :: dy = 0.01_8
#endif

END MODULE OPS_CONSTANTS
