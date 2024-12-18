
! Auto-generated at 2024-12-18 22:20:10.454134 by ops-translator
!
!               ^
!               |        top
!               |    000000000000   r
!               | l  000000000000   i
!   y-direction | e  000000000000   g
!    (index-j)  | f  000000000000   h
!               | t  000000000000   t
!               |       bottom
!               o------------------>
!                      x-direction
!                       (index-i)

PROGRAM laplace
  USE ops_fortran_declarations
  USE ops_fortran_rt_support
  USE set_zero_kernel_module
  USE left_bndcon_kernel_module
  USE right_bndcon_kernel_module
  USE apply_stencil_kernel_module
  USE copy_kernel_module
  !    use OPS_Fortran_hdf5_Declarations
  USE OPS_CONSTANTS

  USE, INTRINSIC :: ISO_C_BINDING

  IMPLICIT NONE

  ! max iterations
  INTEGER, PARAMETER :: iter_max = 100
  INTEGER :: i, j, iter

  REAL(KIND = 8), PARAMETER :: tol = 1.0E-6_8
  REAL(KIND = 8) :: err_diff

  ! integer references (valid inside the OPS library) for ops_block
  TYPE(ops_block) :: grid2D

  !ops_dats
  TYPE(ops_dat) :: d_A, d_Anew

  ! vars for stencils
  INTEGER(KIND = 4) :: s2D_00(2) = [0, 0]
  TYPE(ops_stencil) :: S2D_0pt

  INTEGER(KIND = 4) :: s2D_05(10) = [0, 0, 1, 0, - 1, 0, 0, 1, 0, - 1]
  TYPE(ops_stencil) :: S2D_5pt

  !vars for reduction
  TYPE(ops_reduction) :: h_err
  REAL(KIND = 8) :: error

  INTEGER(KIND = 4) :: d_p(2) = [1, 1]
  !max boundary depths for the dat in the possitive direction
  INTEGER(KIND = 4) :: d_m(2) = [- 1, - 1]
  !max boundary depths for the dat in the negative direction

  !size for OPS
  INTEGER(KIND = 4) :: size(2)

  !base
  INTEGER(KIND = 4) :: base(2) = [1, 1]
  !this is in fortran indexing - start from 1

  !null array - for declaring ops dat
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: temp

  ! profiling
  REAL(KIND = c_double) :: startTime = 0
  REAL(KIND = c_double) :: endTime = 0

  ! iteration range - needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index

  INTEGER :: bottom_range(4), top_range(4), left_range(4), right_range(4), interior_range(4)

  !initialize and declare constants
  imax = 4094
  jmax = 4094
  !    pi = 2.0_8*asin(1.0_8)
#ifdef OPS_WITH_CUDAFOR
  imax_opsconstant = imax
  jmax_opsconstant = jmax
  pi_opsconstant = pi
#endif

  size(1) = jmax
  size(2) = imax

  !                         x(min,max)  y(min,max)
  bottom_range = [0, imax + 1, 0, 0]
  top_range = [0, imax + 1, jmax + 1, jmax + 1]
  left_range = [0, 0, 0, jmax + 1]
  right_range = [imax + 1, imax + 1, 0, jmax + 1]
  interior_range = [1, imax, 1, jmax]

  !-----------------------OPS Initialization------------------------
  CALL ops_init(2)

  !-----------------------OPS Declarations--------------------------

  !declare block
  CALL ops_decl_block(2, grid2D, "grid2D")

  !declare stencils
  CALL ops_decl_stencil(2, 1, s2D_00, S2D_0pt, "0pt_stencil")
  CALL ops_decl_stencil(2, 5, s2D_05, S2D_5pt, "5pt_stencil")

  !declare data on blocks

  !declare ops_dat
  CALL ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_A, "real(kind=8)", "A")
  CALL ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_Anew, "real(kind=8)", "Anew")

  !declare OPS constants
  !CALL ops_decl_const("imax", 1, "integer(kind=4)", imax)
  !CALL ops_decl_const("jmax", 1, "integer(kind=4)", jmax)
  !CALL ops_decl_const("pi", 1, "real(kind=8)", pi)
#ifdef OPS_WITH_OMPOFFLOADFOR
!$OMP TARGET UPDATE TO(imax)
!$OMP TARGET UPDATE TO(jmax)
#endif


  !declare reduction handles
  error = 1.0_8
  CALL ops_decl_reduction_handle(8, h_err, "real(kind=8)", "err")

  ! start timer
  CALL ops_timers(startTime)

  CALL ops_partition("")

  CALL set_zero_kernel_host("set zero", grid2D, 2, bottom_range, &
ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

  CALL set_zero_kernel_host("set zero", grid2D, 2, top_range, &
ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

  CALL left_bndcon_kernel_host("left_bndcon", grid2D, 2, left_range, &
ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_idx())

  CALL right_bndcon_kernel_host("right_bndcon", grid2D, 2, right_range, &
ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_idx())

    IF (ops_is_root() == 1) THEN
    WRITE(*, '(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', imax + 2, ' x', jmax + 2, ' mesh'
  END IF

  iter = 0

  CALL set_zero_kernel_host("set zero", grid2D, 2, bottom_range, &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

  CALL set_zero_kernel_host("set zero", grid2D, 2, top_range, &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

  CALL left_bndcon_kernel_host("left_bndcon", grid2D, 2, left_range, &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_idx())

  CALL right_bndcon_kernel_host("right_bndcon", grid2D, 2, right_range, &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_idx())

    !    call ops_fetch_block_hdf5_file(grid2D, "A.h5")
    !    call ops_fetch_dat_hdf5_file(d_A, "A.h5")

    !    call ops_print_dat_to_txtfile(d_A, "data_A.txt")
    !    call ops_print_dat_to_txtfile(d_Anew, "data_Anew.txt")

    DO WHILE (iter < iter_max .AND. error > tol)

    CALL apply_stencil_kernel_host("apply_stencil", grid2D, 2, interior_range, &
ops_arg_dat(d_A, 1, S2D_5pt, "real(kind=8)", OPS_READ), &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_reduce(h_err, 1, "real(kind=8)", OPS_MAX))
    CALL ops_reduction_result(h_err, error)

    CALL copy_kernel_host("copy", grid2D, 2, interior_range, &
ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_READ))

      IF (MOD(iter, 10) == 0 .AND. ops_is_root() == 1) THEN
      WRITE(*, '(i5,a,f16.7)') iter, ', ', error
    END IF

    iter = iter + 1

  END DO
  ! End of do while loop

    IF (ops_is_root() == 1) THEN
    WRITE(*, '(i5,a,f16.7)') iter, ', ', error
  END IF

  err_diff = ABS((100.0 * (error / 2.421354960840227E-03)) - 100.0)

  WRITE(*, '(a,e18.5,a)') 'Total error is within ', err_diff, ' % of the expected error'

    IF (err_diff .LT. 0.001_8) THEN
    WRITE(*, '(a)') 'This run is considered PASSED'
  ELSE
    WRITE(*, '(a)') 'This test is considered FAILED'
  END IF

  CALL ops_timers(endTime)
  CALL ops_timing_output
  IF (ops_is_root() == 1) THEN
    WRITE(*, '(a,f16.7,a)') ' completed in ', endTime - startTime, ' seconds'
  END IF
  CALL ops_exit

END PROGRAM laplace