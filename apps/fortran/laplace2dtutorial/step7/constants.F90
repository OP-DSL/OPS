MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
    use cudafor
    integer, constant :: imax_opsconstant
    integer, constant :: jmax_opsconstant
    real(8), constant :: pi_opsconstant
    integer :: imax, jmax
    real(8) :: pi = 2.0_8*asin(1.0_8)
#else
    integer :: imax, jmax
    real(8) :: pi = 2.0_8*asin(1.0_8)
#endif

END MODULE OPS_CONSTANTS
