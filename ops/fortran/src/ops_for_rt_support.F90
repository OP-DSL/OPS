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

!> @file
!! @brief OPS fortran back-end library functions declarations
!! @author Gihan Mudalige
!! @details Defines the interoperable data types between OPS-C and OPS-Fortran
!! and the Fortran interface for OPS declaration functions
!!


module OPS_Fortran_RT_Support

  use, intrinsic :: ISO_C_BINDING

#ifdef OPS_WITH_CUDAFOR
  use cudafor
#endif

  !#################################################
  ! Fortran interfaces for ops declaration routines
  ! - binds *_c routines to ops C backend routines
  !#################################################

  interface

  subroutine ops_partition_c (routine) BIND(C,name='ops_partition')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    character(kind=c_char) :: routine(*)
  end subroutine


  subroutine ops_halo_exchanges_c (args, argsNumber, range) BIND(C,name='ops_halo_exchanges')

      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations

      integer(kind=c_int), value :: argsNumber ! number of ops_dat arguments to ops_par_loop
      type(ops_arg), dimension(*) :: args       ! array with ops_args
      integer(4), dimension(*), intent(in), target :: range ! iteration range to determin if halo exchanges are needed

  end subroutine


  subroutine ops_H_D_exchanges_host (args, argsNumber) BIND(C,name='ops_H_D_exchanges_host')
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      integer(kind=c_int), value :: argsNumber ! number of ops_dat arguments to ops_par_loop
      type(ops_arg), dimension(*) :: args       ! array with ops_args
  end subroutine

  subroutine ops_H_D_exchanges_device (args, argsNumber) BIND(C,name='ops_H_D_exchanges_device')
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      integer(kind=c_int), value :: argsNumber ! number of ops_dat arguments to ops_par_loop
      type(ops_arg), dimension(*) :: args       ! array with ops_args
  end subroutine

  subroutine ops_set_dirtybit_host (args, argsNumber) BIND(C,name='ops_set_dirtybit_host')
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      integer(kind=c_int), value :: argsNumber ! number of ops_dat arguments to ops_par_loop
      type(ops_arg), dimension(*) :: args       ! array with ops_args
  end subroutine

  subroutine ops_set_dirtybit_device (args, argsNumber) BIND(C,name='ops_set_dirtybit_device')
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      integer(kind=c_int), value :: argsNumber ! number of ops_dat arguments to ops_par_loop
      type(ops_arg), dimension(*) :: args       ! array with ops_args
  end subroutine

  subroutine ops_set_halo_dirtybit3 (arg, range) BIND(C,name='ops_set_halo_dirtybit3')
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      type(ops_arg) :: arg
      integer(4), dimension(*), intent(in), target :: range ! iteration range to determin if halo exchanges are needed
  end subroutine


  integer(kind=c_int) function ops_is_root () BIND(C,name='ops_is_root')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
  end function

  type(c_ptr) function getDatSizeFromOpsArg (arg) BIND(C,name='getDatSizeFromOpsArg')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
  end function

  integer function getDatDimFromOpsArg (arg) BIND(C,name='getDatDimFromOpsArg')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
  end function

  integer function getDatBaseFromOpsArg1D (arg, start, dim) BIND(C,name='getDatBaseFromOpsArg1D')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
    integer(4), dimension(*), intent(in), target :: start
    integer(kind=c_int), value, intent(in)    :: dim
  end function

  integer function getDatBaseFromOpsArg2D (arg, start, dim) BIND(C,name='getDatBaseFromOpsArg2D')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
    integer(4), dimension(*), intent(in), target :: start
    integer(kind=c_int), value, intent(in)    :: dim
  end function

  integer function getDatBaseFromOpsArg3D (arg, start, dim) BIND(C,name='getDatBaseFromOpsArg3D')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
    integer(4), dimension(*), intent(in), target :: start
    integer(kind=c_int), value, intent(in)    :: dim
  end function

  integer function getDatBaseFromOpsArg3DAMR (arg, start, dim, amrblock) BIND(C,name='getDatBaseFromOpsArg3DAMR')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
    integer(4), dimension(*), intent(in), target :: start
    integer(kind=c_int), value, intent(in)    :: dim
    integer(kind=c_int), value, intent(in)    :: amrblock
  end function

  type(c_ptr) function getReductionPtrFromOpsArg_c (arg, block) BIND(C,name='getReductionPtrFromOpsArg')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
    type(c_ptr), value, intent(in)           :: block
  end function

  type(c_ptr) function getGblPtrFromOpsArg (arg) BIND(C,name='getGblPtrFromOpsArg')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg) :: arg
  end function

  integer function getRange_c (block, start, end, range) BIND(C,name='getRange')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(c_ptr), value, intent(in)           :: block
    type(c_ptr), value :: start
    type(c_ptr), value :: end
    type(c_ptr), intent(in), value           :: range
  end function getRange_c

  integer function getRange2_c (args, nargs, block, start, end, range, arg_idx) BIND(C,name='getRange2')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(ops_arg), dimension(*) :: args       ! array with ops_args
    integer(kind=c_int), value :: nargs ! number of ops_dat arguments to ops_par_loop
    type(c_ptr), value, intent(in)           :: block
    type(c_ptr), value :: start
    type(c_ptr), value :: end
    type(c_ptr), intent(in), value           :: range
    type(c_ptr), intent(in), value           :: arg_idx
  end function getRange2_c

  subroutine getIdx_c (block, start, idx) BIND(C,name='getIdx')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    type(c_ptr), value, intent(in)           :: block
    type(c_ptr), intent(in), value :: start
    type(c_ptr), value :: idx
  end subroutine getIdx_c

!#ifdef OPS_WITH_CUDAFOR
  integer function getOPS_instance::getOPSInstance()->OPS_block_size_x ( ) BIND(C,name='getOPS_instance::getOPSInstance()->OPS_block_size_x')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
  end function

  integer function getOPS_instance::getOPSInstance()->OPS_block_size_y ( ) BIND(C,name='getOPS_instance::getOPSInstance()->OPS_block_size_y')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
  end function
!#endif

  subroutine ops_compute_transfer(ndims, starti, endi, arg, amount) BIND(C,name='ops_compute_transfer_f')
    
      use, intrinsic :: ISO_C_BINDING
      use OPS_Fortran_Declarations
      
      integer(kind=c_int), value :: ndims
      integer(4), dimension(*), intent(in), target :: starti
      integer(4), dimension(*), intent(in), target :: endi
      type(ops_arg) :: arg
      real(kind=c_float) :: amount
  end subroutine ops_compute_transfer

  subroutine setKernelTime (id, name, kernelTime, mpiTime, transfer, count) BIND(C,name='setKernelTime')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value :: id
      character(kind=c_char) :: name(*)
      real(kind=c_double), value :: kernelTime
      real(kind=c_double), value :: mpiTime
      real(kind=c_float), value :: transfer
      integer(kind=c_int), value :: count

  end subroutine setKernelTime

  end interface


  !###################################################################
  ! Fortran subroutines that gets called by an OPS Fortran application
  ! - these calls the relevant *_c routine internally where the *_c
  ! routine is bound to the OPS C backend's actual implemented routine
  !###################################################################

  contains

  subroutine ops_partition (routine)
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    character(kind=c_char,len=*) :: routine
    call ops_partition_c (routine//C_NULL_CHAR)
  end subroutine

  integer function getRange(block, start, end, range )
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    type(ops_block), intent(in)  :: block
    integer(4), dimension(*),target :: start
    integer(4), dimension(*),target :: end
    integer(4), dimension(*), intent(in), target :: range

    getRange = getRange_c ( block%blockCptr, c_loc(start), c_loc(end), c_loc(range))
  end function

  integer function getRange2(args, nargs, block, start, end, range, arg_idx)
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    type(ops_arg), dimension(nargs)  :: args
    integer(kind=c_int), value  :: nargs
    type(ops_block), intent(in)  :: block
    integer(4), dimension(*),target :: start
    integer(4), dimension(*),target :: end
    integer(4), dimension(*), intent(in), target :: range
    integer(4), dimension(*), intent(in), target :: arg_idx

    getRange2 = getRange2_c ( args, nargs, block%blockCptr, c_loc(start), c_loc(end), c_loc(range), c_loc(arg_idx))
  end function

  subroutine getIdx(block, start, idx )
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    type(ops_block), intent(in)  :: block
    integer(4), dimension(*), intent(in), target :: start
    integer(4), dimension(*), target :: idx

    call getIdx_c ( block%blockCptr, c_loc(start), c_loc(idx))
  end subroutine

  type(c_ptr) function getReductionPtrFromOpsArg(arg, block)
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    type(ops_block), intent(in)  :: block
    type(ops_arg), intent(in)  :: arg

    getReductionPtrFromOpsArg =  getReductionPtrFromOpsArg_c ( arg, block%blockCptr)

  end function getReductionPtrFromOpsArg

  subroutine ops_halo_exchanges (args, nargs, range)
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
    implicit none
    integer(4), dimension(*):: range
    integer(kind=c_int), value  :: nargs
    type(ops_arg), dimension(nargs)  :: args

    !range(1) = range(1) - 1
    !range(3) = range(3) - 1
    call ops_halo_exchanges_c (args,nargs,range)
    !range(1) = range(1) + 1
    !range(3) = range(3) + 1

  end subroutine

end module OPS_Fortran_RT_Support
