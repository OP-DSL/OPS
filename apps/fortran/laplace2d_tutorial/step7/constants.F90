MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
    use cudafor
    integer, constant :: imax_OPS
    integer, constant :: jmax_OPS
    real(8), constant :: pi_OPS
    integer :: imax, jmax
    real(8) :: pi
#else
    integer :: imax, jmax
    real(8) :: pi
#endif

END MODULE OPS_CONSTANTS
