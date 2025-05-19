! Open source copyright declaration based on BSD open source template:
! http://www.opensource.org/licenses/bsd-license.php
!
! This file is part of the OPS distribution.
!
! Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
! the main source directory for a full list of copyright holders.
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
! Redistributions of source code must retain the above copyright
! notice, this list of conditions and the following disclaimer.
! Redistributions in binary form must reproduce the above copyright
! notice, this list of conditions and the following disclaimer in the
! documentation and/or other materials provided with the distribution.
! The name of Mike Giles may not be used to endorse or promote products
! derived from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
! WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
! DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
! (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
! ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!

! @brief test application for multi-dimensionsl ops_dats functionality
! @author Gihan Mudalige



program MULTIDIM
  use OPS_Fortran_Reference
  use OPS_Fortran_hdf5_Declarations
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

  implicit none

  intrinsic :: sqrt, real
  !initialize sizes using global values
  integer(kind=4) :: x_cells = 4
  integer(kind=4) :: y_cells = 4

  ! integer(kind=4) :: references (valid inside the OPS library) for ops_block
  type(ops_block)   :: grid2D

  !ops_dats
  type(ops_dat)     :: dat0, dat1

  ! vars for stencils
  integer(kind=4) :: s2D_00_arry(2) = [0,0]
  type(ops_stencil) :: S2D_00

  !vars for reduction
  real(kind=8), dimension(2) :: reduct_result, gbl_test
  type(ops_reduction) :: reduct_dat1

  ! vars for halo_depths
  integer(kind=4) :: d_p(2) = [1,1]   !max halo depths for the dat in the possitive direction
  integer(kind=4) :: d_m(2) = [-1,-1] !max halo depths for the dat in the negative direction

  !size
  integer(kind=4) :: size(2) = [4,4] !size of the dat -- should be identical to the block on which its define on

  !base
  integer(kind=4) :: base1(2) = [1,1] ! this is in fortran indexing
  integer(kind=4) :: base2(2) = [1,1] ! this is in fortran indexing

  !null array
  real(kind=8), dimension(:), allocatable :: temp

  ! profiling
  real(kind=c_double) :: startTime = 0
  real(kind=c_double) :: endTime = 0

  !iteration range
  !iterange needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index
  integer(kind=4) :: iter_range(4) = [1,4,1,4]

  !for validation
  real(kind=8) :: qa_diff

  !-------------------------- Initialisation --------------------------

  !OPS initialisation
  call ops_init(2)
  call ops_set_soa(1)

  !----------------------------OPS Declarations------------------------

  !declare block
  call ops_decl_block(2, grid2D, "grid2D")

  !declare stencils
  call ops_decl_stencil( 2, 1, s2D_00_arry, S2D_00, "00");

  !declare data on blocks
  !declare ops_dat with dim = 2
  call ops_decl_dat(grid2D, 2, size, base1, d_m, d_p, temp,  dat0, "real(kind=8)", "dat0")
  call ops_decl_dat(grid2D, 2, size, base2, d_m, d_p, temp,  dat1, "real(kind=8)", "dat1")

  !initialize and declare constants
  const1 = 5.44_8
  call ops_decl_const("const1", 1, "real(kind=8)", const1)

  !declare reduction handles
  reduct_result(1) = 0.0_8
  reduct_result(2) = 0.0_8
  gbl_test(1) = 1.0_8
  gbl_test(2) = 2.0_8
  call ops_decl_reduction_handle(16, reduct_dat1, "real(kind=8)", "reduct_dat1");

  !decompose the block
  call ops_partition("2D_BLOCK_DECOMPSE")
  call ops_diagnostic_output()
  ! start timer
  call ops_timers ( startTime )

  call ops_par_loop(multidim_kernel, "multidim_kernel", grid2D, 2, iter_range, &
               & ops_arg_dat(dat0, 2, S2D_00, "real(kind=8)", OPS_WRITE), &
               & ops_arg_idx())

  call ops_par_loop(multidim_copy_kernel, "multidim_copy_kernel", grid2D, 2, iter_range, &
               & ops_arg_dat(dat0, 2, S2D_00, "real(kind=8)", OPS_READ), &
               & ops_arg_dat(dat1, 2, S2D_00, "real(kind=8)", OPS_WRITE))

  call ops_par_loop(multidim_print_kernel,"multidim_print_kernel", grid2D, 2, iter_range, &
               & ops_arg_dat(dat0, 2, S2D_00, "real(kind=8)", OPS_READ))

  call ops_par_loop(multidim_reduce_kernel,"multidim_reduce_kernel", grid2D, 2, iter_range, &
               & ops_arg_dat(dat1, 2, S2D_00, "real(kind=8)", OPS_READ), &
               & ops_arg_gbl(gbl_test, 2, "real(kind=8)", OPS_READ),     &
               & ops_arg_reduce(reduct_dat1, 2, "real(kind=8)", OPS_INC))

  call ops_reduction_result(reduct_dat1, reduct_result)

  call ops_timers ( endTime )
  !call ops_print_dat_to_txtfile(dat1, "multidim.dat")
  !call ops_print_dat_to_txtfile(dat0, "multidim.dat")

  call ops_fetch_block_hdf5_file(grid2D, "multidim.h5")
  call ops_fetch_dat_hdf5_file(dat0, "multidim.h5")
  call ops_fetch_dat_hdf5_file(dat1, "multidim.h5")


  !call ops_timing_output (6) ! where is this printing to ? .. problem in what stdout is in fortran
  if (ops_is_root() .eq. 1) then

    write (*,'(a,f16.7,a)') 'Max total runtime =', endTime - startTime,' seconds'

    qa_diff=ABS((100.0_8*((reduct_result(1)+reduct_result(2))/(2*40.00000_8)))-100.0_8)
    write(*,'(a,f16.7,f16.7)') "Reduction result = ", reduct_result
    write(*,'(a,e16.7,a)') "Reduction result is within ",qa_diff,"% of the expected result"

    IF(qa_diff.LT.0.0000000000001) THEN
      write(*,'(a)')"This test is considered PASSED"
    ELSE
      write(*,'(a)')"This test is considered FAILED"
    ENDIF

  end if

  call ops_exit( )

end program MULTIDIM
