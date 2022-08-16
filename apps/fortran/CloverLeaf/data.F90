MODULE DATA_MODULE

    use, intrinsic :: ISO_C_BINDING         
    IMPLICIT NONE

    REAL(KIND=8), PARAMETER :: g_version = 1.0
    INTEGER, PARAMETER :: g_ibig = 640000
    
    ! for file input/output
    integer(kind=c_int) :: g_in, g_out 

    INTEGER :: step

    REAL(KIND=8) :: time, dtold, dtinit

END MODULE DATA_MODULE     
