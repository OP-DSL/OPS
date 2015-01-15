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

  integer(kind=c_int) function ops_is_root () BIND(C,name='ops_is_root')
    use, intrinsic :: ISO_C_BINDING
    use OPS_Fortran_Declarations
  end function

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



end module OPS_Fortran_RT_Support