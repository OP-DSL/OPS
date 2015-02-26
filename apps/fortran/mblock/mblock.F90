!
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

! @Test application for multi-block functionality
! @author Gihan Mudalige, Istvan Reguly

program MBLOCK
  use OPS_Fortran_Reference
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

  implicit none

  !ops blocks
  type(ops_block) :: grid0, grid1

  ! vars for stencils
  integer S2D_00_array(2) /0,0/
  type(ops_stencil) :: S2D_00

  !ops_dats
  type(ops_dat) :: data0, data1

  ! vars for halo_depths
  integer d_p(2) /2,2/   !max halo depths for the dat in the possitive direction
  integer d_m(2) /-2,-2/ !max halo depths for the dat in the negative direction

  !base
  integer base(2) /1,1/ ! this is in fortran indexing

  !size
  integer size(2) /20,20/ !size of the dat

  !null array
  !real(kind=c_double) temp[allocatable](:)
  real(kind=c_double), dimension(:), allocatable :: temp

  !block-holos
  type(ops_halo) :: h0, h1
  type(ops_halo) grp(2)

  !block-holo groups
  type(ops_halo_group) :: halos0, halos1, halos2, halos3, halos4

  !halo vars
  integer halo_iter(2), base_from(2), base_to(2), dir(2), dir_to(2)

  !iteration range
  !iterange needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index
  integer iter_range(4)


  !-------------------------- Initialisation --------------------------

  ! OPS initialisation
  call ops_init(2)

  !----------------------------OPS Declarations------------------------

  ! declare block
  call ops_decl_block(2, grid0, "grid0")
  call ops_decl_block(2, grid1, "grid1")

  ! declare stencils
  call ops_decl_stencil( 2, 1, S2D_00_array, S2D_00, "00")

  call ops_decl_dat(grid0, 1, size, base, d_m, d_p, temp, data0, "double", "data0")
  call ops_decl_dat(grid1, 1, size, base, d_m, d_p, temp, data1, "double", "data1")

  ! straightforward matching orientation halos data0 - data1 in x
  ! last two x lines of data0 and first two of data1
  ! ops_halo_group halos0
  halo_iter(1) = 2
  halo_iter(2) = 20
  base_from(1) = 18
  base_from(2) = 0
  base_to(1) = -2
  base_to(2) = 0
  dir(1) = 1
  dir(2) = 2
  call ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir, h0)
  base_from(1) = 0
  base_to(1) = 20
  call ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir, dir, h1)
  grp(1) = h0
  grp(2) = h1
  call ops_decl_halo_group(2,grp, halos0)


  ! straightforward matching orientation halos data0 - data1 in y
  ! last two y lines of data0 and first two of data1
  ! ops_halo_group halos1
  halo_iter(1) = 20
  halo_iter(2) = 2
  base_from(1) = 0
  base_from(2) = 18
  base_to(1) = 0
  base_to(2) = -2
  dir(1) = 1
  dir(2) = 2
  call ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir, h0)
  base_from(2) = 0
  base_to(2) = 20
  call ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir, dir, h1)
  grp(1) = h0
  grp(2) = h1
  call ops_decl_halo_group(2,grp,halos1)

  ! reverse data1 - data0 in x
  ! last two x lines of data0 and first two of data1, but data1 is flipped in y
  ! ops_halo_group halos2
  halo_iter(1) = 2
  halo_iter(2) = 20
  base_from(1) = 0
  base_from(2) = 0
  base_to(1) = 20
  base_to(2) = 0
  dir(1) = 1
  dir(2) = 2
  dir_to(1) = 1
  dir_to(2) = -2
  call ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to, h0)
  base_from(1) = 18
  base_to(1) = -2
  call ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir,h1)
  grp(1) = h0
  grp(2) = h1
  call ops_decl_halo_group(2,grp,halos2)

  ! reverse data1 - data0 in y
  ! last two y lines of data0 and first two of data1, but data1 is flipped in x
  ! ops_halo_group halos3
  halo_iter(1) = 20
  halo_iter(2) = 2
  base_from(1) = 0
  base_from(2) = 0
  base_to(1) = 0
  base_to(2) = 20
  dir(1) = 1
  dir(2) = 2
  dir_to(1) = -1
  dir_to(2) = 2
  call ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to, h0)
  base_from(2) = 18
  base_to(2) = -2
  call ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir, h1)
  grp(1) = h0
  grp(2) = h1
  call ops_decl_halo_group(2,grp,halos3)

  ! rotated data0-data1 x<->y
  ! last two x lines of data0 to first two y lines of data1 (and back)
  ! ops_halo_group halos4
  halo_iter(1) = 2
  halo_iter(2) = 20
  base_from(1) = 18
  base_from(2) = 0
  base_to(1) = 0
  base_to(2) = -2
  dir(1) = 2
  dir(2) = 1
  dir_to(1) = 2
  dir_to(2) = 1
  call ops_decl_halo(data0, data1, halo_iter, base_from, base_to, dir, dir_to, h0)
  base_from(1) = 0
  base_to(1) = 20
  base_to(2) = 0
  call ops_decl_halo(data1, data0, halo_iter, base_from, base_to, dir_to, dir, h1)
  grp(1) = h0
  grp(2) = h1
  call ops_decl_halo_group(2,grp,halos4)






  call ops_partition("1D_BLOCK_DECOMPOSE")

  !-------------------------- Computations --------------------------

  ! populate
  iter_range(1) =  1
  iter_range(2) =  20
  iter_range(3) =  1
  iter_range(4) =  20
  call ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid0, 2, iter_range, &
              & ops_arg_dat(data0, 1, S2D_00, "real(8)", OPS_WRITE), &
              & ops_arg_idx())

  call ops_par_loop(mblock_populate_kernel, "mblock_populate_kernel", grid1, 2, iter_range, &
              & ops_arg_dat(data1, 1, S2D_00, "real(8)", OPS_WRITE),  &
              & ops_arg_idx())

  !call ops_print_dat_to_txtfile(data1, "data1.txt")
  call ops_halo_transfer(halos0)

  !call ops_halo_transfer(halos1)
  !call ops_halo_transfer(halos2)
  !call ops_halo_transfer(halos3)
  !call ops_halo_transfer(halos4)

  call ops_print_dat_to_txtfile(data0, "data0.txt")
  call ops_print_dat_to_txtfile(data1, "data1.txt")

  call ops_exit( )
end program MBLOCK