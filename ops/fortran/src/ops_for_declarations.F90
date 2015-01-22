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
    integer(kind=c_int) :: elem_size;  ! number of bytes per grid point
    type(c_ptr)         :: data        ! data on host
#ifdef OPS_WITH_CUDAFOR
    type(c_devptr)      :: data_d      ! data on device
#else
    type(c_ptr)         :: data_d      ! data on device
#endif
    type(c_ptr)         :: size        ! size of the array in each block dimension -- including halo
    type(c_ptr)         :: base        ! base offset to 0,0,... from the start of each dimension
    type(c_ptr)         :: d_m         ! halo depth in each dimension, negative direction (at 0 end)
    type(c_ptr)         :: d_p         ! halo depth in each dimension, positive direction (at size end)
    type(c_ptr)         :: name        ! name if the dat
    type(c_ptr)         :: type        ! data type
    integer(kind=c_int) :: dirty_hd    ! flag to indicate dirty status on host and device
    integer(kind=c_int) :: user_managed! indicates whether the user is managing memory
    integer(kind=c_int) :: e_dat       ! is this an edge dat?
  end type ops_dat_core

  type :: ops_dat
      type (ops_dat_core), pointer :: dataPtr => null()
      type (c_ptr)                 :: dataCptr
      integer (kind=c_int)         :: status = -1
  end type ops_dat

  type, BIND(C) :: ops_stencil_core
    integer(kind=c_int) :: index        ! index
    integer(kind=c_int) :: dims         ! dimensionality of the stencil
    type(c_ptr)         :: name         ! name of stencil
    integer(kind=c_int) :: points       ! number of stencil elements
    type(c_ptr)         :: stencil      ! elements in the stencil
    type(c_ptr)         :: stride       ! stride of the stencil
  end type ops_stencil_core

  type :: ops_stencil
    type (ops_stencil_core), pointer :: stencilPtr => null()
    type (c_ptr)                     :: stencilCptr
  end type ops_stencil

  type, BIND(C) :: ops_arg
    type(c_ptr)         :: dat          ! dat
    type(c_ptr)         :: stencil      ! the stencil
    integer(kind=c_int) :: dim          ! dimension of data
    type(c_ptr)         :: data         ! data on host
    type(c_ptr)         :: data_d       ! data on device (for CUDA)
    integer(kind=c_int) :: acc          ! access type
    integer(kind=c_int) :: argtype      ! arg type
    integer(kind=c_int) :: opt          ! falg to indicate whether this is an optional arg, 0 - optional, 1 - not optional
  end type ops_arg

  type, BIND(C) :: ops_reduction_core
    type(c_ptr)         :: data         ! The data
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

    subroutine ops_exit_c (  ) BIND(C,name='ops_exit')
      use, intrinsic :: ISO_C_BINDING
    end subroutine ops_exit_c

    type(c_ptr) function ops_decl_block_c ( dims, name ) BIND(C,name='ops_decl_block')

      use, intrinsic :: ISO_C_BINDING

      import :: ops_block_core

      integer(kind=c_int), value, intent(in)    :: dims
      character(kind=c_char,len=1), intent(in)  :: name(*)
    end function ops_decl_block_c

    type(c_ptr) function ops_decl_dat_c ( block, dim, size, base, d_m, d_p, data, type_size, type, name ) BIND(C,name='ops_decl_dat_char')

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

    end function ops_decl_dat_c

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

    type(c_ptr) function ops_decl_strided_stencil_c ( dims, points, sten, stride, name ) BIND(C,name='ops_decl_strided_stencil')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value               :: dims, points
      type(c_ptr), intent(in), value           :: sten
      type(c_ptr), intent(in), value           :: stride
      character(kind=c_char,len=1), intent(in) :: name(*)

    end function ops_decl_strided_stencil_c


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


    subroutine ops_reduction_result_c (handle, type_size, var) BIND(C,name='ops_reduction_result_char')
      use, intrinsic      :: ISO_C_BINDING
      import :: ops_reduction
      type(c_ptr), value, intent(in)        :: handle
      type(c_ptr)                           :: var
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

    subroutine ops_timing_output (file) BIND(C,name='ops_timing_output')
      use, intrinsic             :: ISO_C_BINDING
      integer(kind=c_int), value :: file
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
    module procedure ops_decl_dat_real_8, ops_decl_dat_integer_4
  end interface ops_decl_dat

  interface ops_reduction_result
    module procedure ops_reduction_result_scalar_real_8, ops_reduction_result_scalar_int_4
  end interface ops_reduction_result



  !###################################################################
  ! Fortran subroutines that gets called by an OPS Fortran application
  ! - these calls the relevant *_c routine internally where the *_c
  ! routine is bound to the OPS C backend's actual implemented routine
  !###################################################################

  contains

    subroutine ops_init ( diags )
      integer(4) :: diags
      integer(4) :: argc = 0
      call ops_init_c ( argc, C_NULL_PTR, diags )
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
    call c_f_pointer ( block%blockCPtr, block%blockPtr )

  end subroutine ops_decl_block

  subroutine ops_decl_stencil ( dims, points, stencil_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_stencil_c ( dims, points, c_loc ( stencil_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCPtr, stencil%stencilPtr)

  end subroutine ops_decl_stencil

  subroutine ops_decl_strided_stencil ( dims, points, stencil_data, stride_data, stencil, name )

    integer, intent(in) :: dims, points
    integer(4), dimension(*), intent(in), target :: stencil_data
    integer(4), dimension(*), intent(in), target :: stride_data
    type(ops_stencil) :: stencil
    character(kind=c_char,len=*):: name

    stencil%stencilCPtr = ops_decl_strided_stencil_c ( dims, points, c_loc ( stencil_data ), c_loc ( stride_data ), name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_stencil variable
    call c_f_pointer (stencil%stencilCPtr, stencil%stencilPtr)

  end subroutine ops_decl_strided_stencil

  subroutine ops_decl_dat_real_8 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size
    integer(4), dimension(*),             target :: base
    integer(4), dimension(*), intent(in), target :: d_m
    integer(4), dimension(*), intent(in), target :: d_p
    real(8), dimension(*), intent(in), target    :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    integer d;
    DO d = 1,2
      base(d) = base(d)-1
    end DO

    dat%dataCPtr = ops_decl_dat_c ( block%blockCPtr, dim, c_loc(size), c_loc(base), c_loc(d_m), c_loc(d_p), c_loc ( data ), 8, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_dat variable
    call c_f_pointer ( dat%dataCPtr, dat%dataPtr )

  end subroutine ops_decl_dat_real_8

  subroutine ops_decl_dat_integer_4 ( block, dim, size, base, d_m, d_p, data, dat, typ, name )

    type(ops_block), intent(in)                  :: block
    integer, intent(in)                          :: dim
    integer(4), dimension(*), intent(in), target :: size
    integer(4), dimension(*),             target :: base
    integer(4), dimension(*), intent(in), target :: d_m
    integer(4), dimension(*), intent(in), target :: d_p
    integer(4), dimension(*), intent(in), target :: data
    type(ops_dat)                                :: dat
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    integer d;
    DO d = 1,2
      base(d) = base(d)-1
    end DO

    dat%dataCPtr = ops_decl_dat_c ( block%blockCPtr, dim, c_loc(size), c_loc(base), c_loc(d_m), c_loc(d_p), c_loc ( data ), 4, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_dat variable
    call c_f_pointer ( dat%dataCPtr, dat%dataPtr )

  end subroutine ops_decl_dat_integer_4

  subroutine ops_decl_reduction_handle ( size, handle, typ, name )

    integer, intent(in)                          :: size
    type(ops_reduction)                          :: handle
    character(kind=c_char,len=*)                 :: name
    character(kind=c_char,len=*)                 :: typ

    handle%reductionCPtr = ops_decl_reduction_handle_c (size, typ//C_NULL_CHAR, name//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the ops_reduction variable
    call c_f_pointer ( handle%reductionCPtr, handle%reductionPtr )

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
    ops_arg_dat = ops_arg_dat_c ( dat%dataCPtr, dim, sten%stencilCPtr, type, access-1 )

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
    ops_arg_dat_opt = ops_arg_dat_opt_c ( dat%dataCPtr, dim, sten%stencilCPtr, type, access-1, flag )

  end function ops_arg_dat_opt

  type(ops_arg) function ops_arg_idx()
    use, intrinsic :: ISO_C_BINDING
    implicit none
    ops_arg_idx = ops_arg_idx_c ()
  end function ops_arg_idx


  type(ops_arg) function ops_arg_reduce(handle, dim, type, access)
    use, intrinsic :: ISO_C_BINDING
    implicit none
    type(ops_reduction) :: handle
    integer(kind=c_int) :: dim
    character(kind=c_char,len=*) :: type
    integer(kind=c_int) :: access
    ! warning: access is in FORTRAN style, while the C style is required here
    ops_arg_reduce= ops_arg_reduce_c( handle%reductionCptr , dim, type, access-1 )

  end function ops_arg_reduce

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
    character(len=*) :: fileName
    call ops_print_dat_to_txtfile_c (dat%dataPtr, fileName)
  end subroutine ops_print_dat_to_txtfile

  subroutine ops_print_dat_to_txtfile_core (dat, fileName)
    type(ops_dat) :: dat
    character(len=*) :: fileName
    call ops_print_dat_to_txtfile_core_c (dat%dataPtr, fileName)
  end subroutine ops_print_dat_to_txtfile_core

 subroutine ops_reduction_result_scalar_real_8 (reduction_handle, var)
    use, intrinsic :: ISO_C_BINDING
    type(ops_reduction) :: reduction_handle
    real(8), target    :: var
    call ops_reduction_result_c (reduction_handle%reductionCptr, 8, c_loc(var))
  end subroutine ops_reduction_result_scalar_real_8

 subroutine ops_reduction_result_scalar_int_4 (reduction_handle, var)
    use, intrinsic :: ISO_C_BINDING
    type(ops_reduction) :: reduction_handle
    integer(4), target  :: var
    call ops_reduction_result_c (reduction_handle%reductionCptr, 4, c_loc(var))
  end subroutine ops_reduction_result_scalar_int_4


 !ops_decl_const -- various versions .. no-ops in ref ?

end module OPS_Fortran_Declarations