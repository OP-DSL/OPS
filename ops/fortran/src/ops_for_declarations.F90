/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief ops fortran back-end library functions declarations
  * @author Gihan Mudalige
  * @details Defines the interoperable data types between OPS-C and OPS-Fortran
  * and the Fortran interface for OPS declaration functions
  */


module OPS_Fortran_Declarations

  use, intrinsic :: ISO_C_BINDING
#ifdef OPS_WITH_CUDAFOR
  use cudafor
#endif

!################################################
! Inteoperable data types for in ops_lib_core.h
!################################################

type, BIND(C) :: ops_block_core
  integer(kind=c_int) :: index        ! index
  integer(kind=c_int) :: dims         ! dimension of vlock, 2D, 3D .. etc
  type(c_ptr)         :: name         ! name if the block
end type ops_block_core

type :: ops_block
  type (ops_block_core), pointer :: setPtr => null()
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
    type (ops_dat_core), pointer :: setPtr => null()
    type (c_ptr)                 :: datCptr
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
  type(c_ptr)         :: dataset      ! dataset
  type(c_ptr)         :: stencil      ! the stencil
  integer(kind=c_int) :: dim          ! dimension of data
  type(c_ptr)         :: data         ! data on host
  type(c_ptr)         :: data_d       ! data on device (for CUDA)
  integer(kind=c_int) :: acc;         ! access type
  integer(kind=c_int) :: argtype      ! arg type
  integer(kind=c_int) :: opt          ! falg to indicate whether this is an optional arg, 0 - optional, 1 - not optional
end type ops_arg



!#################################################
! Fortran interfaces for ops declaration routines
! - binds *_c routines to ops C backend routines
!#################################################

!##################################################################
! Fortran interfaces for different sized ops declaration routines
!##################################################################

!###################################################################
! Fortran subroutines that gets called by an OPS Fortran application
! - these calls the relevant *_c routine internally where the *_c
! routine is bound to the OPS C backend's actual implemented routine
!###################################################################



end module OPS_Fortran_Declarations