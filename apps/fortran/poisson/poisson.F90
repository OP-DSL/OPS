!
! Open source copyright declaration based on BSD open source template:
! http://www.opensource.org/licenses/bsd-license.php
!
! This file is part of the OPS distribution.
!
! Copyright (c) 2015, Mike Giles and others. Please see the AUTHORS file in
! the main source directory for a full list of copyright holders.
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
! ! Redistributions of source code must retain the above copyright
! notice, this list of conditions and the following disclaimer.
! ! Redistributions in binary form must reproduce the above copyright
! notice, this list of conditions and the following disclaimer in the
! documentation and/or other materials provided with the distribution.
! ! The name of Mike Giles may not be used to endorse or promote products
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

! @Test application for multi-block functionality
! @author Gihan Mudalige, Istvan Reguly
!

program POISSON
  use OPS_Fortran_Reference
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

  implicit none

  integer logical_size_x
  integer logical_size_y
  integer ngrid_x
  integer ngrid_y
  integer n_iter

  !ops blocks
  type(ops_block), dimension(:), allocatable :: blocks

  ! vars for stencils
  integer S2D_00_array(2) /0,0/
  type(ops_stencil) :: S2D_00
  integer S2D_00_P10_M10_0P1_0M1_array(10) /0,0, 1,0, -1,0, 0,1, 0,-1/
  type(ops_stencil) :: S2D_00_P10_M10_0P1_0M1

  !ops_reduction
  type(ops_reduction) :: red_err

  !ops_dats
  type(ops_dat),dimension(:), allocatable :: coordx, coordy, u, u2, f, ref

  ! vars for halo_depths
  integer d_p(2) /1,1/   !max halo depths for the dat in the possitive direction
  integer d_m(2) /-1,-1/ !max halo depths for the dat in the negative direction

  !base
  integer base(2) /1,1/ ! this is in fortran indexing

  !size
  integer uniform_size(2)
  integer size(2) !size of the dat

  !null array
  real(8), dimension(:), allocatable :: temp

  !halo vars
  integer(4), dimension(:), allocatable :: sizes, disps
  integer halo_iter(2), base_from(2), base_to(2), dir(2), dir_to(2)

  integer i,j
  character(len=20) buf

  ! constants
  dx = 0.01_8
  dy = 0.01_8

  ! sizes
  logical_size_x =200
  logical_size_y =200
  ngrid_x= 1
  ngrid_y= 1
  n_iter = 10000

  ALLOCATE(blocks(ngrid_x*ngrid_y))

  !-------------------------- Initialisation --------------------------

  ! OPS initialisation
  call ops_init(2)

  !----------------------------OPS Declarations------------------------

  ! declare blocks
  DO j=1,ngrid_y
    DO i=1,ngrid_x
    write(buf,"(A5,I2,A1,I2)") "block",i,",",j
    call ops_decl_block(2, blocks(i+ngrid_x*j), buf)
    END DO
  END DO

  ! declare stencils
  call ops_decl_stencil( 2, 1, S2D_00_array, S2D_00, "00")
  call ops_decl_stencil( 2, 1, S2D_00_P10_M10_0P1_0M1_array, S2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1")

  ! reduction handle for rms variable
  call ops_decl_reduction_handle(8, red_err, "double", "err")

  ! declare dats
  d_p(1) = 1
  d_p(2) = 1
  d_m(1) = -1
  d_m(2) = -1
  base(1) = 0
  base(2) = 0
  uniform_size(1) = (logical_size_x-1)/ngrid_x+1
  uniform_size(2) = (logical_size_y-1)/ngrid_y+1

  ALLOCATE(coordx(ngrid_x*ngrid_y))
  ALLOCATE(coordy(ngrid_x*ngrid_y))
  ALLOCATE(u(ngrid_x*ngrid_y))
  ALLOCATE(u2(ngrid_x*ngrid_y))
  ALLOCATE(f(ngrid_x*ngrid_y))
  ALLOCATE(ref(ngrid_x*ngrid_y))

  ALLOCATE(sizes(2*ngrid_x*ngrid_y))
  ALLOCATE(disps(2*ngrid_x*ngrid_y))

  DO j=1,ngrid_y
    DO i=1,ngrid_x
    size(1) = uniform_size(1)
    size(2) = uniform_size(2)
    if ((i+1)*size(1)>logical_size_x) then
      size(1) = logical_size_x - i*size(1)
    end if
    if ((j+1)*size(2)>logical_size_y) then
      size(2) = logical_size_y - j*size(2)
    end if

    write(buf,"(A6,I2,A1,I2)") "coordx",i,",",j
    call ops_decl_dat(blocks(i+ngrid_x*j), 1, size, base, d_m, d_p, temp, coordx(i+ngrid_x*j), "double", buf)

    END DO
  END DO




  call ops_exit( )
end program POISSON