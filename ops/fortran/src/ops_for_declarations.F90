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

! @brief ops fortran back-end library functions declarations
! @author Gihan Mudalige
! @details Defines the interoperable data types between OPS-C and OPS-Fortran
! and the Fortran interface for OPS declaration functions
!


module OPS_Fortran_Declarations

  use, intrinsic :: ISO_C_BINDING
#ifdef OPS_WITH_CUDAFOR
  use cudafor
#endif

  ! accessing operation codes
  integer(c_int) :: OPS_READ = 1
  integer(c_int) :: OPS_WRITE = 2
  integer(c_int) :: OPS_RW = 3
  integer(c_int) :: OPS_INC = 4
  integer(c_int) :: OPS_MIN = 5
  integer(c_int) :: OPS_MAX = 6

!################################################
! Inteoperable data types for in ops_lib_core.h
!################################################

  type, BIND(C) :: ops_block_core
    integer(kind=c_int) :: index        ! index
    integer(kind=c_int) :: dims         ! dimension of vlock, 2D, 3D .. etc
    type(c_ptr)         :: name         ! name if the block
  end type ops_block_core

  type :: ops_block
    type (ops_block_core), pointer :: blockPtr => null()
    type (c_ptr)                   :: blockCptr
  end type ops_block

  type, BIND(C)         :: ops_dat_core
    integer(kind=c_int) :: index       ! index
    type(c_ptr)         :: block       ! block on which data is defined
    integer(kind=c_int) :: dims        ! number of elements per grid point
    integer(kind=c_int) :: type_size;  ! bytes per primitive = elem_size/dim
    integer(kind=c_int) :: elem_size;  ! number of bytes per grid point
    type(c_ptr)         :: size        ! size of the array in each block dimension -- including halo
    type(c_ptr)         :: base        ! base offset to 0,0,... from the start of each dimension
    type(c_ptr)         :: d_m         ! halo depth in each dimension, negative direction (at 0 end)
    type(c_ptr)         :: d_p         ! halo depth in each dimension, positive direction (at size end)
    type(c_ptr)         :: data        ! data on host
#ifdef OPS_WITH_CUDAFOR
    type(c_devptr)      :: data_d      ! data on device
#else
    type(c_ptr)         :: data_d      ! data on device
#endif
    type(c_ptr)         :: name        ! name if the dat
    type(c_ptr)         :: type        ! data type
    integer(kind=c_int) :: dirty_hd    ! flag to indicate dirty status on host and device
    integer(kind=c_int) :: user_managed! indicates whether the user is managing memory
    integer(kind=c_int) :: is_hdf5     ! indicates whether the user is managing memory
    type(c_ptr)         :: hdf5_file   ! name of the hdf5 file from which this dataset was read
    integer(kind=c_int) :: e_dat       ! is this an edge dat?
    integer(kind=c_long):: mem         ! memory in bytes allocated to this dat
    integer(kind=c_long):: base_offset ! computed, offset in bytes to base index
    integer(kind=c_int) :: amr         ! flag indicating whether AMR dataset
    type(c_ptr)         :: stride      ! stride[*] > 1 if this dat is a coarse dat under multi-grid
  end type ops_dat_core

  type :: ops_dat
      type (ops_dat_core), pointer :: dataPtr => null()
      type (c_ptr)                 :: dataCptr
      integer (kind=c_int)         :: status = -1
  end type ops_dat

  type, BIND(C) :: ops_stencil_core
    integer(kind=c_int) :: index        ! index
    integer(kind=c_int) :: dims         ! dimensionality of the stencil
    integer(kind=c_int) :: points       ! number of stencil elements
    type(c_ptr)         :: name         ! name of stencil
    type(c_ptr)         :: stencil      ! elements in the stencil
    type(c_ptr)         :: stride       ! stride of the stencil
    type(c_ptr)         :: mgrid_stride ! stride of the stencil under multi-grid
    integer(kind=c_int) :: type         ! 0 for regular, 1 for prolongate, 2 for restrict
  end type ops_stencil_core

  type :: ops_stencil
    type (ops_stencil_core), pointer :: stencilPtr => null()
    type (c_ptr)                     :: stencilCptr
  end type ops_stencil

  type, BIND(C) :: ops_arg
    type(c_ptr)         :: dat          ! dat
    type(c_ptr)         :: stencil      ! the stencil
    integer(kind=c_int) :: field        ! field of multi-dimensional data accessed
    integer(kind=c_int) :: dim          ! dimension of data
    type(c_ptr)         :: data         ! data on host
#ifdef OPS_WITH_CUDAFOR
    type(c_devptr)      :: data_d       ! data on device (for CUDA)
#else
    type(c_ptr)         :: data_d
#endif

    integer(kind=c_int) :: acc          ! access type
    integer(kind=c_int) :: argtype      ! arg type
    integer(kind=c_int) :: opt          ! falg to indicate whether this is an optional arg, 0 - optional, 1 - not optional
  end type ops_arg

  type, BIND(C) :: ops_reduction_core
    type(c_ptr)           :: data       ! The data
    integer(kind=c_int) :: size         ! size of data in bytes
    integer(kind=c_int) :: initialized  ! flag indicating whether data has been initialized
    integer(kind=c_int) :: index        ! unique identifier
    integer(kind=c_int) :: acc          ! Type of reduction it was used for last time
    integer(kind=c_int) :: type         ! Type
    type(c_ptr)         :: name         ! Name
  end type ops_reduction_core

  type :: ops_reduction
    type (ops_reduction_core), pointer :: reductionPtr => null()
    type (c_ptr)                       :: reductionCptr
  end type ops_reduction

  type, BIND(C) :: ops_halo_core
    type(c_ptr)         :: from
    type(c_ptr)         :: to
    type(c_ptr)         :: iter_size
    type(c_ptr)         :: from_base
    type(c_ptr)         :: to_base
    type(c_ptr)         :: from_dir
    type(c_ptr)         :: to_dir
    integer(kind=c_int) :: index
  end type ops_halo_core

  type :: ops_halo
      type (ops_halo_core), pointer :: haloPtr => null()
      type (c_ptr)                 :: haloCptr
  end type ops_halo


  type, BIND(C) :: ops_halo_group_core
    integer             :: nhalos
    type(c_ptr)         :: halos
    !type (ops_halo), pointer :: halos => null()
    !type (ops_halo), dimension(*) :: halos
    integer             :: index
  end type ops_halo_group_core

  type :: ops_halo_group
      type (ops_halo_group_core), pointer :: halogroupPtr => null()
      type (c_ptr)                 :: halogroupCptr
  end type ops_halo_group



  !#################################################
  ! Fortran interfaces for ops declaration routines
  ! - binds *_c routines to ops C backend routines
  !#################################################

  interface

    subroutine ops_init_c ( argc, argv, diags ) BIND(C,name='ops_init')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: argc
      type(c_ptr), intent(in)                :: argv
      integer(kind=c_int), intent(in), value :: diags
    end subroutine ops_init_c

    subroutine ops_set_args_c ( argc, argv ) BIND(C,name='ops_set_args')
      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), intent(in), value :: argc
      character(len=1, kind=C_CHAR) :: argv
    end subroutine ops_set_args_c

    subroutine ops_exit_c (  ) BIND(C,name='ops_exit')
      use, intrinsic :: ISO_C_BINDING
    end subroutine ops_exit_c

    type(c_ptr) function ops_decl_block_c ( dims, name ) BIND(C,name='ops_decl_block')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_block_core

      integer(kind=c_int), value, intent(in)    :: dims
      character(kind=c_char,len=1), intent(in)  :: name(*)
    end function ops_decl_block_c

    type(c_ptr) function ops_decl_dat_c ( block, dim, size, base, d_m, d_p, stride, data, type_size, type, name ) &
        & BIND(C,name='ops_decl_dat_char')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_block_core, ops_dat_core

      type(c_ptr), value, intent(in)           :: block
      integer(kind=c_int), value               :: dim, type_size
      character(kind=c_char,len=1), intent(in) :: type(*)
      type(c_ptr), intent(in), value           :: stride
      type(c_ptr), intent(in), value           :: data
      type(c_ptr), intent(in), value           :: size
      type(c_ptr), intent(in), value           :: base
      type(c_ptr), intent(in), value           :: d_m
      type(c_ptr), intent(in), value           :: d_p
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_dat_c

    type(c_ptr) function ops_decl_amrdat_c ( block, dim, size, base, d_m, d_p, data, type_size, type, name ) &
        & BIND(C,name='ops_decl_amrdat_char')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_block_core, ops_dat_core

      type(c_ptr), value, intent(in)           :: block
      integer(kind=c_int), value               :: dim, type_size
      character(kind=c_char,len=1), intent(in) :: type(*)
      type(c_ptr), intent(in), value           :: data
      type(c_ptr), intent(in), value           :: size
      type(c_ptr), intent(in), value           :: base
      type(c_ptr), intent(in), value           :: d_m
      type(c_ptr), intent(in), value           :: d_p
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_amrdat_c

    type(c_ptr) function ops_decl_halo_c ( from, to, iter_size, from_base, to_base, from_dir, to_dir) &
        & BIND(C,name='ops_decl_halo_convert')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_block_core, ops_dat_core, ops_halo_core

      type(c_ptr), value, intent(in)           :: from
      type(c_ptr), value, intent(in)           :: to
      type(c_ptr), intent(in), value           :: iter_size
      type(c_ptr), intent(in), value           :: from_base
      type(c_ptr), intent(in), value           :: to_base
      type(c_ptr), intent(in), value           :: from_dir
      type(c_ptr), intent(in), value           :: to_dir

    end function ops_decl_halo_c

    type(c_ptr) function ops_decl_halo_group_c ( nhalos, halos) BIND(C,name='ops_decl_halo_group')
      use, intrinsic :: ISO_C_BINDING
      import :: ops_halo
      integer(kind=c_int), value               :: nhalos
      type(c_ptr), value, intent(in)           :: halos
      !type(ops_halo), dimension(nhalos)      :: halos
    end function ops_decl_halo_group_c

    type(c_ptr) function ops_decl_halo_group_elem_c ( nhalos, halos, group) BIND(C,name='ops_decl_halo_group_elem')
      use, intrinsic :: ISO_C_BINDING
      import :: ops_halo
      integer(kind=c_int), value               :: nhalos
      type(c_ptr), value, intent(in)           :: halos
      type(c_ptr), value :: group
      !type(ops_halo), dimension(nhalos)      :: halos
    end function ops_decl_halo_group_elem_c


    subroutine ops_halo_transfer_c (group) BIND(C,name='ops_halo_transfer')
      use, intrinsic :: ISO_C_BINDING
      import :: ops_halo_group_core, ops_halo_core
      type(c_ptr), value, intent(in)        :: group
    end subroutine ops_halo_transfer_c

    type(c_ptr) function ops_decl_reduction_handle_c ( size, type, name ) BIND(C,name='ops_decl_reduction_handle')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_reduction_core

      integer(kind=c_int), value               :: size
      character(kind=c_char,len=1), intent(in) :: type(*)
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_reduction_handle_c

    type(c_ptr) function ops_decl_stencil_c ( dims, points, sten, name ) BIND(C,name='ops_decl_stencil')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value               :: dims, points
      type(c_ptr), intent(in), value           :: sten
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_stencil_c

    type(c_ptr) function ops_decl_restrict_stencil_c ( dims, points, sten, stride, name ) BIND(C,name='ops_decl_restrict_stencil')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value               :: dims, points
      type(c_ptr), intent(in), value           :: sten
      type(c_ptr), intent(in), value           :: stride
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_restrict_stencil_c

    type(c_ptr) function ops_decl_prolong_stencil_c ( dims, points, sten, stride, name ) BIND(C,name='ops_decl_prolong_stencil')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value               :: dims, points
      type(c_ptr), intent(in), value           :: sten
      type(c_ptr), intent(in), value           :: stride
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_prolong_stencil_c

    type(c_ptr) function ops_decl_strided_stencil_c ( dims, points, sten, stride, name ) BIND(C,name='ops_decl_strided_stencil')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value               :: dims, points
      type(c_ptr), intent(in), value           :: sten
      type(c_ptr), intent(in), value           :: stride
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_strided_stencil_c

    subroutine ops_free_dat_c( dat ) BIND(C,name='ops_free_dat')
      use, intrinsic :: ISO_C_BINDING
      type(c_ptr), value, intent(in) :: dat
    end subroutine

    function ops_arg_dat_c ( dat, dim, sten, type, acc ) BIND(C,name='ops_arg_dat')

      use, intrinsic :: ISO_C_BINDING
      import :: ops_arg

      type(ops_arg)                  :: ops_arg_dat_c
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value     :: dim
      type(c_ptr), value, intent(in) :: sten
      character(kind=c_char,len=1)   :: type(*)
      integer(kind=c_int), value     :: acc

    end function ops_arg_dat_c

    function ops_arg_restrict_c ( dat, idx, dim, sten, type, acc ) BIND(C,name='ops_arg_restrict')

      use, intrinsic :: ISO_C_BINDING
      import :: ops_arg

      type(ops_arg)                  :: ops_arg_restrict_c
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value     :: idx, dim
      type(c_ptr), value, intent(in) :: sten
      character(kind=c_char,len=1)   :: type(*)
      integer(kind=c_int), value     :: acc

    end function ops_arg_restrict_c

    function ops_arg_prolong_c ( dat, idx, dim, sten, type, acc ) BIND(C,name='ops_arg_prolong')

      use, intrinsic :: ISO_C_BINDING
      import :: ops_arg

      type(ops_arg)                  :: ops_arg_prolong_c
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value     :: idx, dim
      type(c_ptr), value, intent(in) :: sten
      character(kind=c_char,len=1)   :: type(*)
      integer(kind=c_int), value     :: acc

    end function ops_arg_prolong_c

    function ops_arg_dat2_c ( dat, idx, dim, sten, type, acc ) BIND(C,name='ops_arg_dat2')

      use, intrinsic :: ISO_C_BINDING
      import :: ops_arg

      type(ops_arg)                  :: ops_arg_dat2_c
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value     :: idx, dim
      type(c_ptr), value, intent(in) :: sten
      character(kind=c_char,len=1)   :: type(*)
      integer(kind=c_int), value     :: acc

    end function ops_arg_dat2_c


    function ops_arg_dat_opt_c ( dat, dim, sten, type, acc, flag ) BIND(C,name='ops_arg_dat_opt')

      use, intrinsic :: ISO_C_BINDING
      import :: ops_arg

      type(ops_arg)                  :: ops_arg_dat_opt_c
      type(c_ptr), value, intent(in) :: dat
      integer(kind=c_int), value     :: dim
      type(c_ptr), value, intent(in) :: sten
      character(kind=c_char,len=1)   :: type(*)
      integer(kind=c_int), value     :: acc
      integer(kind=c_int), value     :: flag
    end function ops_arg_dat_opt_c

    function ops_arg_idx_c ( ) BIND(C,name='ops_arg_idx')
      use, intrinsic :: ISO_C_BINDING
      import         :: ops_arg
      type(ops_arg)  :: ops_arg_idx_c
    end function ops_arg_idx_c


    function ops_arg_reduce_c ( handle, dim, type, acc ) BIND(C,name='ops_arg_reduce')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_arg, ops_reduction

      type(ops_arg) :: ops_arg_reduce_c

      type(c_ptr), value, intent(in) :: handle
      integer(kind=c_int), value   :: dim
      character(kind=c_char,len=1) :: type(*)
      integer(kind=c_int), value   :: acc

    end function ops_arg_reduce_c

    function ops_arg_gbl_c ( data, dim, size, acc ) BIND(C,name='ops_arg_gbl_char')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_arg

      type(ops_arg) :: ops_arg_gbl_c

      type(c_ptr), value :: data
      integer(kind=c_int), value :: dim, size
      integer(kind=c_int), value :: acc

    end function ops_arg_gbl_c


    subroutine ops_reduction_result_c (handle, type_size, var) BIND(C,name='ops_reduction_result_char')
      use, intrinsic      :: ISO_C_BINDING
      import :: ops_reduction
      type(c_ptr), value, intent(in)  :: handle
      !type(c_ptr), intent(in)  :: handle
      type(c_ptr) , value       :: var
      integer(kind=c_int), value, intent(in):: type_size
    end subroutine ops_reduction_result_c


    subroutine ops_timers_core_f ( cpu, et ) BIND(C,name='ops_timers_core')
      use, intrinsic      :: ISO_C_BINDING
      real(kind=c_double) :: cpu, et
    end subroutine ops_timers_core_f

    subroutine ops_timers_f ( cpu, et ) BIND(C,name='ops_timers')
      use, intrinsic      :: ISO_C_BINDING
      real(kind=c_double) :: cpu, et
    end subroutine ops_timers_f

    subroutine ops_timing_output () BIND(C,name='ops_timing_output_stdout')
      use, intrinsic             :: ISO_C_BINDING
    end subroutine ops_timing_output

    subroutine ops_diagnostic_output ( ) BIND(C,name='ops_diagnostic_output')
      use, intrinsic :: ISO_C_BINDING
    end subroutine ops_diagnostic_output

    subroutine ops_printf_c (line) BIND(C,name='ops_printf')
      use ISO_C_BINDING
      character(kind=c_char) :: line(*)
    end subroutine ops_printf_c

    subroutine ops_fprintf_c (file, line) BIND(C,name='ops_fprintf')
      use ISO_C_BINDING
      integer(kind=c_int), value :: file
      character(kind=c_char)     :: line(*)
    end subroutine ops_fprintf_c

    subroutine ops_print_dat_to_txtfile_c (dat, file_name) BIND(C,name='ops_print_dat_to_txtfile')
      use ISO_C_BINDING
      import :: ops_dat_core
      type(ops_dat_core) :: dat
      character(kind=c_char,len=1), intent(in) :: file_name(*)
    end subroutine ops_print_dat_to_txtfile_c

    subroutine ops_print_dat_to_txtfile_core_c (dat, file_name) BIND(C,name='ops_print_dat_to_txtfile_core')
      use ISO_C_BINDING
      import :: ops_dat_core
      type(ops_dat_core) :: dat
      character(kind=c_char,len=1), intent(in) :: file_name(*)
    end subroutine ops_print_dat_to_txtfile_core_c

  end interface

  !##################################################################
  ! Fortran interfaces for different sized ops declaration routines
  !##################################################################

  interface ops_decl_dat
    module procedure ops_decl_dat_real_8, ops_decl_dat_integer_4, &
    &  ops_decl_dat_real_8_3d, ops_decl_dat_real_8_4d, ops_decl_dat_real_8_5d
  end interface ops_decl_dat

  interface ops_decl_amrdat
     module procedure ops_decl_amrdat_real_8, ops_decl_amrdat_integer_4, &
     &  ops_decl_amrdat_real_8_3d, ops_decl_amrdat_real_8_4d, ops_decl_amrdat_real_8_5d
   end interface ops_decl_amrdat

  interface ops_reduction_result
    module procedure ops_reduction_result_scalar_real_8, ops_reduction_result_scalar_int_4, &
    & ops_reduction_result_real_8
  end interface ops_reduction_result

  interface ops_arg_gbl
    module procedure ops_arg_gbl_real_scalar, ops_arg_gbl_int_scalar, ops_arg_gbl_real_1dim, &
    & ops_arg_gbl_real_2dim, ops_arg_gbl_real_3dim, ops_arg_gbl_real_4dim
  end interface ops_arg_gbl

  !###################################################################
  ! Fortran subroutines that gets called by an OPS Fortran application
  ! - these calls the relevant *_c routine internally where the *_c
  ! routine is bound to the OPS C backend's actual implemented routine
  !###################################################################

  contains

    subroutine ops_init ( diags )
      integer(4) :: diags
      integer(kind=c_int) :: argc
      integer :: i
      character(kind=c_char,len=64)           :: temp

      !Get the command line arguments - needs to be handled using Fortrn
      argc = command_argument_count()

      do i = 1, argc
        call get_command_argument(i, temp)
        call ops_set_args_c (argc, temp) !special function to set args
      end do

      call ops_init_c (0, C_NULL_PTR, diags)

    end subroutine ops_init



  subroutine ops_exit ( )
    call ops_exit_c (  )
  end subroutine ops_exit

  subroutine ops_decl_block ( dims, block, name )

    integer(kind=c_int), value, intent(in) :: dims
    type(ops_block)                        :: block
    character(kind=c_char,len=*)           :: name

    block%blockCPtr = ops_decl_block_c ( dims, name//char(0) )

    ! convert the generated C pointer to Fortran pointer and store it inside the op_block variable
    call c_f_pointer ( block%blockCptr, block%blockPtr )

  end subroutine ops_decl_block

  subroutine ops_decl_stencil ( dims, points, stencil_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_stencil_c ( dims, points, c_loc ( stencil_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCptr, stencil%stencilPtr)

  end subroutine ops_decl_stencil

  subroutine ops_decl_restrict_stencil ( dims, points, stencil_data, stride_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    integer(4), dimension(*), intent(in), target :: stride_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_restrict_stencil_c ( dims, points, c_loc ( stencil_data ), &
     & c_loc ( stride_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCptr, stencil%stencilPtr)

  end subroutine ops_decl_restrict_stencil

  subroutine ops_decl_prolong_stencil ( dims, points, stencil_data, stride_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    integer(4), dimension(*), intent(in), target :: stride_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_prolong_stencil_c ( dims, points, c_loc ( stencil_data ), &
     & c_loc ( stride_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCptr, stencil%stencilPtr)

  end subroutine ops_decl_prolong_stencil

  subroutine ops_decl_strided_stencil ( dims, points, stencil_data, stride_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    integer(4), dimension(*), intent(in), target :: stride_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_strided_stencil_c ( dims, points, c_loc ( stencil_data ), &
     & c_loc ( stride_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCptr, stencil%stencilPtr)

  end subroutine ops_decl_strided_stencil

  subroutine ops_decl_dat_generic( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size
    integer(4), dimension(*),             target :: base
    integer(4), dimension(*), intent(in), target :: d_m
    integer(4), dimension(*), intent(in), target :: d_p
    type(c_ptr), intent(in), value    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ
    integer(4), target                           :: stride(5)

    integer d;
    DO d = 1, block%blockPtr%dims
      base(d) = base(d)-1
      stride(d) = 1
    end DO

    dat%dataCPtr = ops_decl_dat_c ( block%blockCptr, dim, c_loc(size), c_loc(base), c_loc(d_m), &
     & c_loc(d_p), c_loc(stride), data, 8, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    DO d = 1, block%blockPtr%dims
      base(d) = base(d)+1
    end DO

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_dat variable
    call c_f_pointer ( dat%dataCptr, dat%dataPtr )

  end subroutine ops_decl_dat_generic

  subroutine ops_decl_amrdat_generic( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size
    integer(4), dimension(*),             target :: base
    integer(4), dimension(*), intent(in), target :: d_m
    integer(4), dimension(*), intent(in), target :: d_p
    type(c_ptr), intent(in), value    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    integer d;
    DO d = 1, block%blockPtr%dims
      base(d) = base(d)-1
    end DO

    dat%dataCPtr = ops_decl_amrdat_c ( block%blockCptr, dim, c_loc(size), c_loc(base), c_loc(d_m), &
     & c_loc(d_p), data, 8, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    DO d = 1, block%blockPtr%dims
      base(d) = base(d)+1
    end DO

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_dat variable
    call c_f_pointer ( dat%dataCptr, dat%dataPtr )

  end subroutine ops_decl_amrdat_generic

  subroutine ops_decl_dat_real_8 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
    real(8), dimension(*), intent(in), target    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    call ops_decl_dat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

  end subroutine ops_decl_dat_real_8

  subroutine ops_decl_dat_real_8_3d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
    real(8), dimension(:,:,:), intent(in), target    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    call ops_decl_dat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

  end subroutine ops_decl_dat_real_8_3d

  subroutine ops_decl_dat_real_8_4d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
    real(8), dimension(:,:,:,:), intent(in), target    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    call ops_decl_dat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

  end subroutine ops_decl_dat_real_8_4d

  subroutine ops_decl_dat_real_8_5d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
    real(8), dimension(:,:,:,:,:), intent(in), target    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    call ops_decl_dat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

  end subroutine ops_decl_dat_real_8_5d

  subroutine ops_decl_dat_integer_4 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
    integer(4), dimension(*), intent(in), target :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    call ops_decl_dat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

  end subroutine ops_decl_dat_integer_4

   subroutine ops_decl_amrdat_real_8 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

     type(ops_block), intent(in)                  :: block
     integer, intent(in)                          :: dim
     integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
     real(8), dimension(*), intent(in), target    :: data
     type(ops_dat)                                :: dat
     character(kind=c_char,len=*)                 :: name
     character(kind=c_char,len=*)                 :: typ

     call ops_decl_amrdat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

   end subroutine ops_decl_amrdat_real_8

   subroutine ops_decl_amrdat_real_8_3d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

     type(ops_block), intent(in)                  :: block
     integer, intent(in)                          :: dim
     integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
     real(8), dimension(:,:,:), intent(in), target    :: data
     type(ops_dat)                                :: dat
     character(kind=c_char,len=*)                 :: name
     character(kind=c_char,len=*)                 :: typ

     call ops_decl_amrdat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

   end subroutine ops_decl_amrdat_real_8_3d

   subroutine ops_decl_amrdat_real_8_4d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

     type(ops_block), intent(in)                  :: block
     integer, intent(in)                          :: dim
     integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
     real(8), dimension(:,:,:,:), intent(in), target    :: data
     type(ops_dat)                                :: dat
     character(kind=c_char,len=*)                 :: name
     character(kind=c_char,len=*)                 :: typ

     call ops_decl_amrdat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

   end subroutine ops_decl_amrdat_real_8_4d

   subroutine ops_decl_amrdat_real_8_5d ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

     type(ops_block), intent(in)                  :: block
     integer, intent(in)                          :: dim
     integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
     real(8), dimension(:,:,:,:,:), intent(in), target    :: data
     type(ops_dat)                                :: dat
     character(kind=c_char,len=*)                 :: name
     character(kind=c_char,len=*)                 :: typ

     call ops_decl_amrdat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

   end subroutine ops_decl_amrdat_real_8_5d

   subroutine ops_decl_amrdat_integer_4 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

     type(ops_block), intent(in)                  :: block
     integer, intent(in)                          :: dim
     integer(4), dimension(*), intent(in), target :: size,base,d_m,d_p
     integer(4), dimension(*), intent(in), target :: data
     type(ops_dat)                                :: dat
     character(kind=c_char,len=*)                 :: name
     character(kind=c_char,len=*)                 :: typ

     call ops_decl_amrdat_generic(block,dim,size,base,d_m,d_p,c_loc(data),dat,typ,name)

   end subroutine ops_decl_amrdat_integer_4

  subroutine ops_decl_reduction_handle ( size, handle, typ, name )

    integer, intent(in)                          :: size
    type(ops_reduction)                          :: handle
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    handle%reductionCPtr = ops_decl_reduction_handle_c (size, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_reduction variable
    call c_f_pointer ( handle%reductionCptr, handle%reductionPtr )

  end subroutine ops_decl_reduction_handle


  type(ops_arg) function ops_arg_dat(dat, dim, sten, type, access)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_dat) :: dat
    integer(kind=c_int) :: dim
    type(ops_stencil) :: sten
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access

    if (dat%dataPtr%dims .ne. dim) then
      print *, "Wrong dim",dim,dat%dataPtr%dims
    endif
    ! warning: access and idx are in FORTRAN style, while the C style is required here
    ops_arg_dat = ops_arg_dat_c ( dat%dataCptr, dim, sten%stencilCptr, type, access-1 )

  end function ops_arg_dat

  type(ops_arg) function ops_arg_dat_opt(dat, dim, sten, type, access, flag)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_dat) :: dat
    integer(kind=c_int) :: dim, flag
    type(ops_stencil) :: sten
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access

    if (dat%dataPtr%dims .ne. dim) then
      print *, "Wrong dim",dim,dat%dataPtr%dims
    endif
    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_dat_opt = ops_arg_dat_opt_c ( dat%dataCptr, dim, sten%stencilCptr, type, access-1, flag )

  end function ops_arg_dat_opt

  type(ops_arg) function ops_arg_idx()
    use, intrinsic :: ISO_C_BINDING
    implicit none
    ops_arg_idx = ops_arg_idx_c ()
  end function ops_arg_idx


  type(ops_arg) function ops_arg_reduce(handle, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(ops_reduction) :: handle
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_reduce= ops_arg_reduce_c( handle%reductionCptr , dim, typ//C_NULL_CHAR, access-1 )

  end function ops_arg_reduce

  type(ops_arg) function ops_arg_gbl_real_scalar(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    real(8), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_real_scalar = ops_arg_gbl_c( c_loc(data) , dim, 8, access-1 )

  end function ops_arg_gbl_real_scalar

  type(ops_arg) function ops_arg_gbl_int_scalar(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    integer(4), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_int_scalar = ops_arg_gbl_c( c_loc(data) , dim, 4, access-1 )

  end function ops_arg_gbl_int_scalar

  type(ops_arg) function ops_arg_gbl_real_1dim(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    real(8), dimension(*), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_real_1dim = ops_arg_gbl_c( c_loc(data) , dim, 8, access-1 )

  end function ops_arg_gbl_real_1dim

  type(ops_arg) function ops_arg_gbl_real_2dim(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    real(8), dimension(:,:), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_real_2dim = ops_arg_gbl_c( c_loc(data) , dim, 8, access-1 )

  end function ops_arg_gbl_real_2dim

  type(ops_arg) function ops_arg_gbl_real_3dim(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    real(8), dimension(:,:,:), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_real_3dim = ops_arg_gbl_c( c_loc(data) , dim, 8, access-1 )

  end function ops_arg_gbl_real_3dim

  type(ops_arg) function ops_arg_gbl_real_4dim(data, dim, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    real(8), dimension(:,:,:,:), target :: data
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_gbl_real_4dim = ops_arg_gbl_c( c_loc(data) , dim, 8, access-1 )

  end function ops_arg_gbl_real_4dim


  type(ops_arg) function ops_arg_restrict(dat, idx, dim, stencil, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(ops_dat) :: dat
    integer(kind=c_int) :: idx, dim
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_restrict = ops_arg_restrict_c( dat%dataCptr, idx, dim, &
      & stencil%stencilCptr, typ, access-1 )

  end function ops_arg_restrict

  type(ops_arg) function ops_arg_prolong(dat, idx, dim, stencil, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(ops_dat) :: dat
    integer(kind=c_int) :: idx, dim
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_prolong = ops_arg_prolong_c( dat%dataCptr, idx, dim, &
      & stencil%stencilCptr, typ, access-1 )

  end function ops_arg_prolong

  type(ops_arg) function ops_arg_dat2(dat, idx, dim, stencil, typ, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(ops_dat) :: dat
    integer(kind=c_int) :: idx, dim
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*) :: typ
    integer(kind=c_int) :: access

    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_dat2 = ops_arg_dat2_c( dat%dataCptr, idx, dim, &
      & stencil%stencilCptr, typ, access-1 )

  end function ops_arg_dat2

  subroutine ops_decl_halo (from, to, iter_size, from_base, to_base, from_dir, to_dir, halo)

    type(ops_dat)                                :: from
    type(ops_dat)                                :: to
    integer(4), dimension(*), intent(in), target :: iter_size
    integer(4), dimension(*), intent(in), target :: from_base
    integer(4), dimension(*), intent(in), target :: to_base
    integer(4), dimension(*), intent(in), target :: from_dir
    integer(4), dimension(*), intent(in), target :: to_dir
    type(ops_halo)                               :: halo

    halo%haloCptr = ops_decl_halo_c (from%dataCptr, to%dataCptr, c_loc(iter_size), &
      & c_loc(from_base), c_loc(to_base), c_loc(from_dir), c_loc(to_dir))

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_decl_halo variable
    call c_f_pointer ( halo%haloCptr, halo%haloPtr )

  end subroutine ops_decl_halo

  subroutine ops_decl_halo_group (nhalos, halos, group)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    integer(kind=c_int), value                        :: nhalos
    type(ops_halo), dimension(nhalos), target         :: halos
    type(ops_halo_group)                              :: group
    integer i
    type(c_ptr) :: temp = c_null_ptr
    type(c_ptr) :: dummy = c_null_ptr

    !need to call special routine for each halo in the hlos array
    !-- due to issues in passing an array for types in fortran
    DO i = 1, nhalos
      temp = ops_decl_halo_group_elem_c ( nhalos, c_loc(halos(i)), temp)
    END DO

    !special case if nhalos == 0
    if (nhalos .EQ. 0) then
      temp = ops_decl_halo_group_elem_c ( nhalos, dummy, temp)
    end if

    !group%halogroupCptr = ops_decl_halo_group_c ( nhalos, c_loc(halos))
    !group%halogroupCptr = ops_decl_halo_group_c (nhalos, halos)
    group%halogroupCptr = temp
    temp = c_null_ptr
    ! convert the generated C pointer to Fortran pointer and store it inside the ops_halo_group variable
    call c_f_pointer ( group%halogroupCptr, group%halogroupPtr )
  end subroutine ops_decl_halo_group

  subroutine ops_free_dat(dat)
    type(ops_dat) :: dat
    call ops_free_dat_c(dat%dataCptr)
  end subroutine ops_free_dat

  subroutine ops_halo_transfer (group)
      type(ops_halo_group)                :: group
      call ops_halo_transfer_c(group%halogroupCptr)
  end subroutine ops_halo_transfer

  subroutine ops_timers ( et )
    real(kind=c_double) :: et
    real(kind=c_double) :: cpu = 0
    call ops_timers_f (cpu, et)
  end subroutine ops_timers

  subroutine ops_timers_core ( et )
    real(kind=c_double) :: et
    real(kind=c_double) :: cpu = 0
    call ops_timers_core_f (cpu, et)
  end subroutine ops_timers_core

  subroutine ops_printf (line)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    character(kind=c_char,len=*) :: line
    call ops_printf_c (line//C_NULL_CHAR)
  end subroutine

  subroutine ops_fprintf (file, line)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    character(kind=c_char,len=*) :: line
    integer(kind=c_int) :: file
    call ops_fprintf_c (file, line//C_NULL_CHAR)
  end subroutine

  subroutine ops_print_dat_to_txtfile (dat, fileName)
    type(ops_dat) :: dat
    character(kind=c_char,len=*) :: fileName
    call ops_print_dat_to_txtfile_c (dat%dataPtr, fileName//C_NULL_CHAR)
  end subroutine ops_print_dat_to_txtfile

  subroutine ops_print_dat_to_txtfile_core (dat, fileName)
    type(ops_dat) :: dat
    character(kind=c_char,len=*) :: fileName
    call ops_print_dat_to_txtfile_core_c (dat%dataPtr, fileName//C_NULL_CHAR)
  end subroutine ops_print_dat_to_txtfile_core

 subroutine ops_reduction_result_scalar_real_8 (reduction_handle, var)
    use, intrinsic :: ISO_C_BINDING
    type(ops_reduction) :: reduction_handle
    real(8), target    :: var

    call ops_reduction_result_c (reduction_handle%reductionCptr, &
      & reduction_handle%reductionPtr%size, c_loc(var))

  end subroutine ops_reduction_result_scalar_real_8

 subroutine ops_reduction_result_scalar_int_4 (reduction_handle, var)
    use, intrinsic :: ISO_C_BINDING
    type(ops_reduction) :: reduction_handle
    integer(4), target  :: var
    call ops_reduction_result_c (reduction_handle%reductionCptr, &
      & reduction_handle%reductionPtr%size, c_loc(var))
  end subroutine ops_reduction_result_scalar_int_4

 subroutine ops_reduction_result_real_8 (reduction_handle, var)
    use, intrinsic :: ISO_C_BINDING
    type(ops_reduction) :: reduction_handle
    real(8), dimension(*), target :: var

    call ops_reduction_result_c (reduction_handle%reductionCptr, &
      & reduction_handle%reductionPtr%size, c_loc(var))
  end subroutine ops_reduction_result_real_8


 !ops_decl_const -- various versions .. no-ops in ref ?

end module OPS_Fortran_Declarations
