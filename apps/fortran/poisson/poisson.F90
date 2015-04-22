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

! sizes
#define logical_size_x 200
#define logical_size_y 200
#define ngrid_x 2
#define ngrid_y 2
#define n_iter  10000

program POISSON
  use OPS_Fortran_Reference
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

  implicit none

  !integer logical_size_x
  !integer logical_size_y
  !integer ngrid_x
  !integer ngrid_y
  !integer n_iter

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
  integer :: sizes(2*ngrid_x*ngrid_y), disps(2*ngrid_x*ngrid_y)

  integer halo_iter(2), base_from(2), base_to(2), dir(2), dir_to(2)

  !ops blocks
  type(ops_block) :: blocks(ngrid_x*ngrid_y)

  ! vars for stencils
  integer S2D_00_array(2) /0,0/
  type(ops_stencil) :: S2D_00
  integer S2D_00_P10_M10_0P1_0M1_array(10) /0,0, 1,0, -1,0, 0,1, 0,-1/
  type(ops_stencil) :: S2D_00_P10_M10_0P1_0M1

  !ops_reduction
  type(ops_reduction) :: red_err
  real(8) :: err

  !ops_dats
  type(ops_dat) :: coordx(ngrid_x*ngrid_y), coordy(ngrid_x*ngrid_y)
  type(ops_dat) :: u(ngrid_x*ngrid_y), u2(ngrid_x*ngrid_y), f(ngrid_x*ngrid_y), ref(ngrid_x*ngrid_y)

  !ops_halos
  type(ops_halo) :: halos((2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y)))

  !ops_halo group
  type(ops_halo_group) :: u_halos

  !iteration ranges
  integer iter_range(4)

  integer i,j, off, iter
  character(len=20) buf

  ! profiling
  real(kind=c_double) :: startTime = 0
  real(kind=c_double) :: endTime = 0

  ! constants
  dx = 0.01_8
  dy = 0.01_8

  !ALLOCATE(blocks(ngrid_x*ngrid_y))

  !-------------------------- Initialisation --------------------------

  ! OPS initialisation
  call ops_init(2)

  !----------------------------OPS Declarations------------------------

  ! declare blocks
  DO j=1,ngrid_y
    DO i=1,ngrid_x
    write(buf,"(A5,I2,A1,I2)") "block",i,",",j
    call ops_decl_block(2, blocks((i-1)+ngrid_x*(j-1)+1), buf)
    END DO
  END DO

  ! declare stencils
  call ops_decl_stencil( 2, 1, S2D_00_array, S2D_00, "00")
  call ops_decl_stencil( 2, 5, S2D_00_P10_M10_0P1_0M1_array, S2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1")

  ! reduction handle for rms variable
  call ops_decl_reduction_handle(8, red_err, "real(8)", "err")

  ! declare dats
  d_p(1) = 1
  d_p(2) = 1
  d_m(1) = -1
  d_m(2) = -1
  base(1) = 1
  base(2) = 1
  uniform_size(1) = (logical_size_x-1)/ngrid_x+1
  uniform_size(2) = (logical_size_y-1)/ngrid_y+1

  DO j=1,ngrid_y
    DO i=1,ngrid_x
    size(1) = uniform_size(1)
    size(2) = uniform_size(2)
    if ((i)*size(1)>logical_size_x) then
      size(1) = logical_size_x - (i-1)*size(1)
    end if
    if ((j)*size(2)>logical_size_y) then
      size(2) = logical_size_y - (j-1)*size(2)
    end if

    write(buf,"(A6,I2,A1,I2)") "coordx",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, coordx((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)
    write(buf,"(A6,I2,A1,I2)") "coordy",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, coordy((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)
    write(buf,"(A6,I2,A1,I2)") "u",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, u((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)
    write(buf,"(A6,I2,A1,I2)") "u2",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, u2((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)
    write(buf,"(A6,I2,A1,I2)") "f",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, f((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)
    write(buf,"(A6,I2,A1,I2)") "ref",i,",",j
    call ops_decl_dat(blocks((i-1)+ngrid_x*(j-1)+1), 1, size, base, d_m, d_p, temp, ref((i-1)+ngrid_x*(j-1)+1), "real(8)", buf)

    sizes(2*((i-1)+ngrid_x*(j-1))+1) = size(1)
    sizes(2*((i-1)+ngrid_x*(j-1))+2) = size(2)
    disps(2*((i-1)+ngrid_x*(j-1))+1) = (i-1)*uniform_size(1)
    disps(2*((i-1)+ngrid_x*(j-1))+2) = (j-1)*uniform_size(2)

    END DO
  END DO

  !write (*,*) "sizes", sizes
  !write (*,*) "disps", disps

  off = 1
  DO j = 1, ngrid_y
    DO i = 1, ngrid_x
      if ((i-1) > 0) then
      halo_iter(1) = 1
      halo_iter(2) = sizes(2*((i-1)+ngrid_x*(j-1))+2)
      base_from(1) = sizes(2*((i-2)+ngrid_x*(j-1))+1)
      base_from(2) = 1
      base_to(1) = 0
      base_to(2) = 1
      dir(1) = 1
      dir(2) = 2

      !write (*,*) "in first ", i,j, halo_iter
      !write (*,*) "in first ", i,j, base_from
      !write (*,*) "in first ", i,j, base_to

      call ops_decl_halo(u((i-2)+ngrid_x*(j-1)+1), u((i-1)+ngrid_x*(j-1)+1), halo_iter, base_from, base_to, dir, dir, halos(off))
      off = off + 1
      base_from(1) = 1; base_to(1) = sizes(2*((i-1)+ngrid_x*(j-1))+1)+1
      !write (*,*) "in first", i,j, base_from
      !write (*,*) "in first base to", i,j, base_to
      call ops_decl_halo(u((i-1)+ngrid_x*(j-1)+1), u((i-2)+ngrid_x*(j-1)+1), halo_iter, base_from, base_to, dir, dir, halos(off))
      off = off + 1
      end if
      if ((j-1) > 0) then
      halo_iter(1) = sizes(2*((i-1)+ngrid_x*(j-1))+1)
      halo_iter(2) = 1
      base_from(1) = 1
      base_from(2) = sizes(2*((i-1)+ngrid_x*(j-2))+2)
      base_to(1) = 1
      base_to(2) = 0
      dir(1) = 1
      dir(2) = 2

      !write (*,*) "in second", i,j, halo_iter
      !write (*,*) "in second", i,j, base_from
      !write (*,*) "in second", i,j, base_to

      call ops_decl_halo(u((i-1)+ngrid_x*(j-2)+1), u((i-1)+ngrid_x*(j-1)+1), halo_iter, base_from, base_to, dir, dir, halos(off))
      off = off + 1
      base_from(2) = 1; base_to(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1)+1
      !write (*,*) "in second", i,j, base_from
      !write (*,*) "in second base to", i,j, base_to
      call ops_decl_halo(u((i-1)+ngrid_x*(j-1)+1), u((i-1)+ngrid_x*(j-2)+1), halo_iter, base_from, base_to, dir, dir, halos(off))
      off = off + 1
      end if
    end do
  end do
  if ((off-1) .NE. 2*(ngrid_x*(ngrid_y-1)+(ngrid_x-1)*ngrid_y)) then
    write (*,*) "Something is not right"
  end if
  call ops_decl_halo_group((off-1),halos, u_halos)



  call ops_partition("")

  !-------------------------- Computations --------------------------

  ! start timer
  call ops_timers(startTime)

  ! populate forcing, reference solution and boundary conditions
  DO j = 1, ngrid_y
    DO i = 1, ngrid_x
      iter_range(1) = 0
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1) +1
      iter_range(3) = 0
      iter_range(4) = sizes(2*((i-1)+ngrid_x*(j-1))+2) +1
      !write(*,*) iter_range
      call ops_par_loop(poisson_populate_kernel, "poisson_populate_kernel", blocks((i-1)+ngrid_x*(j-1)+1), 2, iter_range, &
            &  ops_arg_gbl(disps(2*((i-1)+ngrid_x*(j-1))+1), 1, "integer(4)", OPS_READ), &
            &  ops_arg_gbl(disps(2*((i-1)+ngrid_x*(j-1))+2), 1, "integer(4)", OPS_READ), &
            &  ops_arg_idx(), &
            &  ops_arg_dat(u((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_WRITE), &
            &  ops_arg_dat(f((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_WRITE), &
            &  ops_arg_dat(ref((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_WRITE))
    END DO
  END DO

  ! initial guess 0
  DO j = 1, ngrid_y
    DO i = 1, ngrid_x
      iter_range(1) = 1
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1)
      iter_range(3) = 1
      iter_range(4) = sizes(2*((i-1)+ngrid_x*(j-1))+2)
      !write(*,*) iter_range
      call ops_par_loop(poisson_initialguess_kernel, "poisson_initialguess_kernel", blocks((i-1)+ngrid_x*(j-1)+1), 2, iter_range, &
                & ops_arg_dat(u((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_WRITE))

    END DO
  END DO

  !
  ! Main iterative loop
  !
  DO iter = 1, n_iter

    call ops_halo_transfer(u_halos)

    DO j = 1, ngrid_y
      DO i = 1, ngrid_x
      iter_range(1) = 1
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1)
      iter_range(1) = 1
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+2)
        call ops_par_loop(poisson_stencil_kernel, "poisson_stencil_kernel", blocks((i-1)+ngrid_x*(j-1)+1), 2, iter_range, &
                & ops_arg_dat(u((i-1)+ngrid_x*(j-1)+1), 1, S2D_00_P10_M10_0P1_0M1, "real(8)", OPS_READ), &
                & ops_arg_dat(f((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_READ), &
                & ops_arg_dat(u2((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_WRITE));
      END DO
    END DO


    DO j = 1, ngrid_y
      DO i = 1, ngrid_x
        iter_range(1) = 1
        iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1)
        iter_range(1) = 1
        iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+2)
        call ops_par_loop(poisson_update_kernel, "poisson_update_kernel", blocks((i-1)+ngrid_x*(j-1)+1), 2, iter_range, &
                & ops_arg_dat(u2((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_READ), &
                & ops_arg_dat(u((i-1)+ngrid_x*(j-1)+1) , 1, S2D_00, "real(8)", OPS_WRITE))
      END DO
    END DO

  END DO

  !call ops_print_dat_to_txtfile(u(1), "poisson.dat")
  !call ops_print_dat_to_txtfile(ref(1), "poisson.dat")
  !call exit()

  err = 0.0_8
  DO j = 1, ngrid_y
    DO i = 1, ngrid_x
      iter_range(1) = 1
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+1)
      iter_range(1) = 1
      iter_range(2) = sizes(2*((i-1)+ngrid_x*(j-1))+2)
      call ops_par_loop(poisson_error_kernel, "poisson_error_kernel", blocks((i-1)+ngrid_x*(j-1)+1), 2, iter_range, &
              & ops_arg_dat(u((i-1)+ngrid_x*(j-1)+1), 1, S2D_00, "real(8)", OPS_READ), &
              & ops_arg_dat(ref((i-1)+ngrid_x*(j-1)+1) , 1, S2D_00, "real(8)", OPS_READ), &
              & ops_arg_reduce(red_err, 1, "real(8)", OPS_INC))
    END DO
  END DO

  call ops_reduction_result(red_err, err)

  call ops_timers(endTime)

  if (ops_is_root() .eq. 1) then
    write (*,*) 'Total error: ', err
  end if

  if (ops_is_root() .eq. 1) then
    write (*,*) 'Max total runtime =', endTime - startTime,'seconds'
  end if

  call ops_exit( )
end program POISSON