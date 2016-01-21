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


module OPS_Fortran_hdf5_Declarations

  use OPS_Fortran_Declarations
  use, intrinsic :: ISO_C_BINDING
#ifdef OPS_WITH_CUDAFOR
  use cudafor
#endif

  interface

	type(c_ptr) function ops_decl_block_hdf5_c (dim, blockName, fileName) BIND(C,name='ops_decl_block_hdf5')

      use, intrinsic :: ISO_C_BINDING
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1), intent(in) :: blockName
      character(kind=c_char,len=1), intent(in) :: fileName

    end function ops_decl_block_hdf5_c

	type(c_ptr) function ops_decl_dat_hdf5_c (blockName, dim, type, datName, fileName) BIND(C,name='ops_decl_dat_hdf5')

      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: blockName
      integer(kind=c_int), value :: dim
      character(kind=c_char,len=1), intent(in) :: type
      character(kind=c_char,len=1), intent(in) :: datName
      character(kind=c_char,len=1), intent(in) :: fileName

    end function ops_decl_dat_hdf5_c

	type(c_ptr) function ops_decl_stencil_hdf5_c (dims, points, stencilName, fileName) BIND(C,name='ops_decl_stencil_hdf5')

      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value :: dims
      integer(kind=c_int), value :: points
      character(kind=c_char,len=1), intent(in) :: stencilName
      character(kind=c_char,len=1), intent(in) :: fileName

    end function ops_decl_stencil_hdf5_c

    type(c_ptr) function ops_decl_strided_stencil_hdf5_c (dims, points, stencilName, fileName) BIND(C,name='ops_decl_stencil_hdf5')
    !check if bind C name is implemented .. need to implement ops_decl_strided_stencil_hdf5
      use, intrinsic :: ISO_C_BINDING

      integer(kind=c_int), value :: dims
      integer(kind=c_int), value :: points
      character(kind=c_char,len=1), intent(in) :: stencilName
      character(kind=c_char,len=1), intent(in) :: fileName


    end function ops_decl_strided_stencil_hdf5_c

	subroutine ops_fetch_dat_hdf5_file_c (dat, fileName) BIND(C,name='ops_fetch_dat_hdf5_file')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: dat
      character(len=1,kind=c_char) :: fileName(*)
    end subroutine ops_fetch_dat_hdf5_file_c

	subroutine ops_fetch_block_hdf5_file_c (block, fileName) BIND(C,name='ops_fetch_block_hdf5_file')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: block
      character(len=1,kind=c_char) :: fileName(*)
    end subroutine ops_fetch_block_hdf5_file_c

    subroutine ops_fetch_stencil_hdf5_file_c (stencil, fileName) BIND(C,name='ops_fetch_stencil_hdf5_file')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: stencil
      character(len=1,kind=c_char) :: fileName(*)
    end subroutine ops_fetch_stencil_hdf5_file_c

    subroutine ops_fetch_halo_hdf5_file_c (halo, fileName) BIND(C,name='ops_fetch_halo_hdf5_file')
      use, intrinsic :: ISO_C_BINDING

      type(c_ptr), value, intent(in)           :: halo
      character(len=1,kind=c_char) :: fileName(*)
    end subroutine ops_fetch_halo_hdf5_file_c

  end interface

contains

  subroutine ops_decl_block_hdf5 (block, dim, blockName, fileName)

    type(ops_block) :: block
    character(kind=c_char,len=*) :: blockName
    integer, intent(in) :: dim
    character(kind=c_char,len=*) :: fileName

    ! assume names are /0 terminated
	block%blockCptr = ops_decl_block_hdf5_c (dim, blockName//C_NULL_CHAR, fileName//C_NULL_CHAR)

    ! convert the generated C pointer to Fortran pointer and store it inside the op_set variable
    call c_f_pointer ( block%blockCPtr, block%blockPtr )

  end subroutine ops_decl_block_hdf5


  subroutine ops_decl_dat_hdf5 ( dat, block, dat_size, type, datName, fileName, status )
    implicit none

    type(ops_dat) :: dat
    type(ops_block), intent(in) :: block
    integer, intent(in) :: dat_size
    character(kind=c_char,len=*) :: type
    character(kind=c_char,len=*) :: datName
    character(kind=c_char,len=*) :: fileName
    integer (kind=c_int) :: status

    status = -1
    dat%dataPtr => null()

    ! assume names are /0 terminated
    dat%dataCPtr = ops_decl_dat_hdf5_c ( block%blockCPtr, dat_size, type//C_NULL_CHAR, datName//C_NULL_CHAR, fileName//C_NULL_CHAR)

    ! convert the generated C pointer to Fortran pointer and store it inside the op_dat variable
    call c_f_pointer ( dat%dataCPtr, dat%dataPtr )
    if (associated(dat%dataPtr)) then
      status = 1 !dat%dataPtr%size
    end if
    ! debugging

  end subroutine ops_decl_dat_hdf5


  subroutine ops_decl_stencil_hdf5 ( stencil, dims, points, stencilName, fileName, status )

    type(ops_stencil) :: stencil
    integer, intent(in) :: dims
    integer, intent(in) :: points
	character(kind=c_char,len=*) :: stencilName
    character(kind=c_char,len=*) :: fileName
    integer (kind=c_int) :: status

    status = -1
    stencil%stencilPtr => null()

    ! assume names are /0 terminated - will fix this if needed later
    stencil%stencilCPtr = ops_decl_stencil_hdf5_c ( mapdims, points, stencilName//C_NULL_CHAR, fileName//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( stencil%stencilCPtr, stencil%stencilPtr )
    if (associated(stencil%stencilPtr)) then
      status = 1
    end if
  end subroutine ops_decl_stencil_hdf5

  subroutine ops_decl_strided_stencil_hdf5 ( stencil, dims, points, stencilName, fileName, status )

    type(ops_stencil) :: stencil
    integer, intent(in) :: dims
    integer, intent(in) :: points
	character(kind=c_char,len=*) :: stencilName
    character(kind=c_char,len=*) :: fileName
    integer (kind=c_int) :: status

    status = -1
    stencil%stencilPtr => null()

    ! assume names are /0 terminated - will fix this if needed later
    stencil%stencilCPtr = ops_decl_strided_stencil_hdf5_c ( dims, points, stencilName//C_NULL_CHAR, fileName//C_NULL_CHAR )

    ! convert the generated C pointer to Fortran pointer and store it inside the op_map variable
    call c_f_pointer ( stencil%stencilCPtr, stencil%stencilPtr )
    if (associated(stencil%stencilPtr)) then
      status = 1
    end if
  end subroutine ops_decl_strided_stencil_hdf5


  subroutine ops_fetch_dat_hdf5_file (dat, file_name)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_dat), intent(in) :: dat
    character(kind=c_char,len=*) :: file_name
    call ops_fetch_dat_hdf5_file_c (dat%dataCPtr, file_name//C_NULL_CHAR)

  end subroutine ops_fetch_dat_hdf5_file

  subroutine ops_fetch_block_hdf5_file (block, file_name)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_block), intent(in) :: block
    character(kind=c_char,len=*) :: file_name

    call ops_fetch_block_hdf5_file_c (block%blockCPtr, file_name//C_NULL_CHAR)

  end subroutine ops_fetch_block_hdf5_file

  subroutine ops_fetch_stencil_hdf5_file (stencil, file_name)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_stencil), intent(in) :: stencil
    character(kind=c_char,len=*) :: file_name

    call ops_fetch_stencil_hdf5_file_c (stencil%stencilCPtr, file_name//C_NULL_CHAR)

  end subroutine ops_fetch_stencil_hdf5_file

  subroutine ops_fetch_halo_hdf5_file (halo, file_name)

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_halo), intent(in) :: halo
    character(kind=c_char,len=*) :: file_name

    call ops_fetch_halo_hdf5_file_c (halo%haloCPtr, file_name//C_NULL_CHAR)

  end subroutine ops_fetch_halo_hdf5_file


!subroutine ops_read_dat_hdf5 ( dat )
!    implicit none

    !type(ops_dat) :: dat

!end subroutine ops_read_dat_hdf5


end module OPS_Fortran_hdf5_Declarations
