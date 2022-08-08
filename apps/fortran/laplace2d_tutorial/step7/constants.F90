MODULE OPS_CONSTANTS

#ifdef OPS_WITH_CUDAFOR
    use cudafor
    integer, constant :: imax_opsconstant
    integer, constant :: jmax_opsconstant
    real(8), constant :: pi_opsconstant
    integer :: imax, jmax
    real(8) :: pi
#else
    integer :: imax, jmax
    real(8) :: pi
#endif

END MODULE OPS_CONSTANTS
