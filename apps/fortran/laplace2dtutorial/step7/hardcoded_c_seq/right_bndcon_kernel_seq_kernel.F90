! Auto-generated at 2024-12-18 11:34:46.158949 by ops-translator

MODULE RIGHT_BNDCON_KERNEL_MODULE

    USE OPS_FORTRAN_DECLARATIONS
    USE OPS_FORTRAN_RT_SUPPORT

    USE OPS_CONSTANTS
    USE, INTRINSIC :: ISO_C_BINDING

    IMPLICIT NONE

    INTERFACE

SUBROUTINE right_bndcon_kernel_host_c(name, args, nargs, index, dim, range, block) BIND(C,name='right_bndcon_kernel_host_c')
    USE, INTRINSIC :: ISO_C_BINDING
    import :: ops_block_core, ops_arg

    character(kind=c_char,len=1) :: name(*)
    type(c_ptr), value           :: args
    integer(kind=c_int), value   :: nargs
    integer(kind=c_int), value   :: index
    integer(kind=c_int), value   :: dim
    type(c_ptr), value      :: range
    type(c_ptr), value      :: block

END SUBROUTINE

    END INTERFACE    

    CONTAINS

SUBROUTINE right_bndcon_kernel_host( userSubroutine, block, dim, range, &
    opsArg1, &
    opsArg2 &
    )

    CHARACTER(KIND=C_CHAR,LEN=*), INTENT(IN), TARGET :: userSubroutine
    TYPE(ops_block), INTENT(IN) :: block
    INTEGER(KIND=4), INTENT(IN) :: dim
    INTEGER(KIND=4), DIMENSION(2*dim), INTENT(INOUT), TARGET :: range
    INTEGER(KIND=4), DIMENSION(2*dim), TARGET :: range_tmp

    TYPE(ops_arg), INTENT(IN) :: opsArg1
    TYPE(ops_arg), INTENT(IN) :: opsArg2

    TYPE(ops_arg), DIMENSION(2), TARGET :: opsArgArray
    INTEGER(KIND=4) :: n_indx
    CHARACTER(LEN=40) :: namelit

    namelit = "right_bndcon_kernel"

    opsArgArray(1) = opsArg1
    opsArgArray(2) = opsArg2

    DO n_indx = 1, 2
        range_tmp(2*n_indx-1) = range(2*n_indx-1)-1
        range_tmp(2*n_indx)   = range(2*n_indx)
    END DO

    CALL right_bndcon_kernel_host_c(namelit//c_null_char, c_loc(opsArgArray), &
                                    2, 3, dim, c_loc(range_tmp), &
                                    block%blockCptr)

END SUBROUTINE

END MODULE RIGHT_BNDCON_KERNEL_MODULE
