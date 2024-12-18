! Auto-generated at 2024-12-18 22:20:10.823195 by ops-translator

MODULE COPY_KERNEL_MODULE

    USE OPS_FORTRAN_DECLARATIONS
    USE OPS_FORTRAN_RT_SUPPORT

    USE OPS_CONSTANTS
    USE, INTRINSIC :: ISO_C_BINDING

    IMPLICIT NONE

    INTEGER(KIND=4) :: xdim1
    INTEGER(KIND=4) :: ydim1
#define OPS_ACC1(x,y) ((x) + (xdim1*(y)) + 1)

    INTEGER(KIND=4) :: xdim2
    INTEGER(KIND=4) :: ydim2
#define OPS_ACC2(x,y) ((x) + (xdim2*(y)) + 1)

    CONTAINS

!   =============
!   User function
!   =============

#ifdef __INTEL_COMPILER
!DEC$ ATTRIBUTES FORCEINLINE :: copy_kernel
#endif
SUBROUTINE copy_kernel(A, Anew)

    REAL(KIND=8), DIMENSION(1), INTENT(IN) :: Anew
    REAL(KIND=8), DIMENSION(1) :: A

    A(OPS_ACC1(0,0)) = Anew(OPS_ACC2(0,0))

END SUBROUTINE

#undef OPS_ACC1
#undef OPS_ACC2

SUBROUTINE copy_kernel_wrap( &
    opsDat1Local, &
    opsDat2Local, &
    dat1_base, &
    dat2_base, &
    start_indx, &
    end_indx )

    REAL(KIND=8), DIMENSION(*), INTENT(OUT) :: opsDat1Local
    INTEGER(KIND=4), INTENT(IN) :: dat1_base

    REAL(KIND=8), DIMENSION(*), INTENT(IN) :: opsDat2Local
    INTEGER(KIND=4), INTENT(IN) :: dat2_base

    INTEGER(KIND=4), DIMENSION(2), INTENT(IN) :: start_indx, end_indx

    INTEGER(KIND=4) :: n_x, n_y

    !$OMP PARALLEL DO PRIVATE(n_x,n_y)
    DO n_y = 1, end_indx(2)-start_indx(2)+1
        !$OMP SIMD
        DO n_x = 1, end_indx(1)-start_indx(1)+1

#ifdef _CRAYFTN
                !DIR$ INLINE
#endif
                CALL copy_kernel( &
                opsDat1Local(dat1_base + ((n_x-1)*1*1) + ((n_y-1)*xdim1*1*1)), &
                opsDat2Local(dat2_base + ((n_x-1)*1*1) + ((n_y-1)*xdim2*1*1)) &
               )

        END DO
        !$OMP END SIMD
    END DO

END SUBROUTINE copy_kernel_wrap

!   ===============
!   Host subroutine
!   ===============
#ifndef OPS_LAZY
SUBROUTINE copy_kernel_host( userSubroutine, block, dim, range, &
    opsArg1, &
    opsArg2 &
    )

    CHARACTER(KIND=C_CHAR,LEN=*), INTENT(IN) :: userSubroutine
    TYPE(ops_block), INTENT(IN) :: block
    INTEGER(KIND=4), INTENT(IN) :: dim
    INTEGER(KIND=4), DIMENSION(2*dim), INTENT(IN) :: range

    TYPE(ops_arg), INTENT(IN) :: opsArg1
    TYPE(ops_arg), INTENT(IN) :: opsArg2

    TYPE(ops_arg), DIMENSION(2) :: opsArgArray

#else
SUBROUTINE copy_kernel_host_execute( descPtr )

    TYPE(ops_kernel_descriptor), INTENT(IN) :: descPtr
    TYPE(ops_block) :: block
    INTEGER(KIND=C_INT) :: dim
    INTEGER(KIND=C_INT), POINTER, DIMENSION(:) :: range
    CHARACTER(KIND=C_CHAR), POINTER, DIMENSION(:) :: userSubroutine
    TYPE(ops_arg), POINTER, DIMENSION(:) :: opsArgArray

    TYPE(ops_arg) :: opsArg1
    TYPE(ops_arg) :: opsArg2

#endif

    REAL(KIND=8), POINTER, DIMENSION(:) :: opsDat1Local
    INTEGER(KIND=4) :: opsDat1Cardinality
    INTEGER(KIND=4), POINTER, DIMENSION(:)  :: dat1_size
    INTEGER(KIND=4) :: dat1_base

    REAL(KIND=8), POINTER, DIMENSION(:) :: opsDat2Local
    INTEGER(KIND=4) :: opsDat2Cardinality
    INTEGER(KIND=4), POINTER, DIMENSION(:)  :: dat2_size
    INTEGER(KIND=4) :: dat2_base

    REAL(KIND=8) :: t1__, t2__, t3__
    REAL(KIND=4) :: transfer_total, transfer

    INTEGER(KIND=4), DIMENSION(2) :: start_indx, end_indx
    INTEGER(KIND=4) :: n_indx
    CHARACTER(LEN=40) :: kernelName

    kernelName = "copy_kernel"

#ifdef OPS_LAZY
!   ==========================
!   Set from kernel descriptor
!   ==========================
    dim = descPtr%dim
    CALL c_f_pointer(descPtr%range, range, [2*dim])
    CALL c_f_pointer(descPtr%name, userSubroutine, [descPtr%name_len])
    block%blockCptr = descPtr%block
    CALL c_f_pointer(block%blockCptr, block%blockPtr)
    CALL c_f_pointer(descPtr%args, opsArgArray, [descPtr%nargs])

    opsArg1 = opsArgArray(1)
    opsArg2 = opsArgArray(2)
#else
    opsArgArray(1) = opsArg1
    opsArgArray(2) = opsArg2
#endif

    CALL setKernelTime(5, kernelName//c_null_char, 0.0_8, 0.0_8, 0.0_4, 1)
    CALL ops_timers_core(t1__)

#if defined(OPS_MPI) && !defined(OPS_LAZY)
    IF ( getRange(block, start_indx, end_indx, range) < 0 ) THEN
        RETURN
    END IF
#elif !defined(OPS_MPI)  && !defined(OPS_LAZY)
    DO n_indx = 1, 2
        start_indx(n_indx) = range(2*n_indx-1)
        end_indx  (n_indx) = range(2*n_indx)
    END DO
#else
    DO n_indx = 1, 2
        start_indx(n_indx) = range(2*n_indx-1) + 1
        end_indx  (n_indx) = range(2*n_indx)
    END DO
#endif

    CALL c_f_pointer(getDatSizeFromOpsArg(opsArg1), dat1_size, [dim])
    xdim1 = dat1_size(1)
    ydim1 = dat1_size(2)
    opsDat1Cardinality = opsArg1%dim * xdim1 * ydim1
    dat1_base = getDatBaseFromOpsArg2D(opsArg1, start_indx, 1)
    CALL c_f_pointer(opsArg1%data, opsDat1Local, [opsDat1Cardinality])

    CALL c_f_pointer(getDatSizeFromOpsArg(opsArg2), dat2_size, [dim])
    xdim2 = dat2_size(1)
    ydim2 = dat2_size(2)
    opsDat2Cardinality = opsArg2%dim * xdim2 * ydim2
    dat2_base = getDatBaseFromOpsArg2D(opsArg2, start_indx, 1)
    CALL c_f_pointer(opsArg2%data, opsDat2Local, [opsDat2Cardinality])

!   ==============
!   Halo exchanges
!   ==============
#ifndef OPS_LAZY
    CALL ops_H_D_exchanges_host(opsArgArray, 2)
    CALL ops_halo_exchanges(opsArgArray, 2, range)
    CALL ops_H_D_exchanges_host(opsArgArray, 2)
#endif

    CALL ops_timers_core(t2__)

!   ==============================
!   Call kernel wrapper subroutine
!   ==============================
    CALL copy_kernel_wrap( &
                        opsDat1Local, &
                        opsDat2Local, &
                        dat1_base, &
                        dat2_base, &
                        start_indx, &
                        end_indx )

    CALL ops_timers_core(t3__)

#ifndef OPS_LAZY
    CALL ops_set_dirtybit_host(opsArgArray, 2)
    CALL ops_set_halo_dirtybit3(opsArg1, range)
#endif

!   ========================
!   Timing and data movement
!   ========================
    transfer_total = 0.0_4
    CALL ops_compute_transfer(2, start_indx, end_indx, opsArg1, transfer)
    transfer_total = transfer_total + transfer
    CALL ops_compute_transfer(2, start_indx, end_indx, opsArg2, transfer)
    transfer_total = transfer_total + transfer

    CALL setKernelTime(5, kernelName//c_null_char, t3__-t2__, t2__-t1__, transfer_total, 0)

END SUBROUTINE

#ifdef OPS_LAZY
SUBROUTINE copy_kernel_host( userSubroutine, block, dim, range, &
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

    namelit = "copy_kernel"

    opsArgArray(1) = opsArg1
    opsArgArray(2) = opsArg2

    DO n_indx = 1, 2
        range_tmp(2*n_indx-1) = range(2*n_indx-1)-1
        range_tmp(2*n_indx)   = range(2*n_indx)
    END DO

    CALL create_kerneldesc_and_enque(namelit//c_null_char, c_loc(opsArgArray), &
                                    2, 5, dim, 0, c_loc(range_tmp), &
                                    block%blockCptr, c_funloc(copy_kernel_host_execute))

END SUBROUTINE
#endif

END MODULE COPY_KERNEL_MODULE
