! @brief SHSGC top level program
! @author Satya P. Jammy, converted to OPS by Gihan Mudalige
! @details
!
! This version is based on Fortran and uses the OPS prototype highlevel domain
! specific API for developing multi-block Structured mesh applications


program SHSGC
  use OPS_Fortran_Reference
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

   implicit none

   intrinsic :: sqrt, real

  !ops blocks
  type(ops_block) :: shsgc_grid

  !ops_dats
  type(ops_dat) :: x
  type(ops_dat) :: rho_old, rho_new, rho_res
  type(ops_dat) :: rhou_old, rhou_new, rhou_res
  type(ops_dat) :: rhov_old, rhov_new
  type(ops_dat) :: rhoE_old, rhoE_new, rhoE_res
  type(ops_dat) :: rhoin
  type(ops_dat) :: r, al, alam, gt, tht, ep2, cmp, cf, eff, s
  type(ops_dat) :: readvar

  ! vars for stencils
  integer S1D_0_array(1) /0/
  integer S1D_01_array(2) /0,1/
  integer S1D_0M1_array(2) /0,-1/
  integer S1D_0M1M2P1P2_array(5) /0,-1,-2,1,2/
  type(ops_stencil) :: S1D_0, S1D_01, S1D_0M1
  type(ops_stencil) :: S1D_0M1M2P1P2

  ! vars for halo_depths
  integer d_p(1) /2/   !max halo depths for the dat in the possitive direction
  integer d_m(1) /-2/ !max halo depths for the dat in the negative direction

  !base
  integer base(1) /1/ ! this is in fortran indexing

  !size
  integer nxp /204/
  integer size(1) /204/ !size of the dat -- should be identical to the block on which its define on

  !null array
  real(8) temp[allocatable](:)

  !iteration range
  !iterange needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index
  integer nxp_range(2)

  !-------------------------- Initialisation --------------------------

  !OPS initialisation
  call ops_init(2)

  !----------------------------OPS Declarations------------------------

  !declare block
  call ops_decl_block(2, shsgc_grid, "shsgc grid")

  !declare stencils
  call ops_decl_stencil( 2, 1, S1D_0_array, S1D_0, "0")
  call ops_decl_stencil( 2, 1, S1D_01_array, S1D_01, "0,1")
  call ops_decl_stencil( 2, 1, S1D_0M1_array, S1D_0M1, "0,-1")
  call ops_decl_stencil( 2, 1, S1D_0M1M2P1P2_array, S1D_0M1M2P1P2, "0,-1,-2,1,2")


  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, x, "real(8)", "x")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_old, "real(8)", "rho_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_new, "real(8)", "rho_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_res, "real(8)", "rho_res")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_old, "real(8)", "rhou_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_new, "real(8)", "rhou_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_res, "real(8)", "rhou_res")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhov_old, "real(8)", "rhov_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhov_new, "real(8)", "rhov_new")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_old, "real(8)", "rhoE_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_new, "real(8)", "rhoE_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_res, "real(8)", "rhoE_res")

  !extra dat for rhoin
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoin, "real(8)", "rhoin");

  !
  ! Initialize with the test case
  !
  nxp_range(1) =  1
  nxp_range(2) =  nxp
  call ops_par_loop(initialize_kernel, "initialize_kernel", shsgc_grid, 1, nxp_range, &
          & ops_arg_dat(x, 1, S1D_0, "real(8)", OPS_WRITE), &
          & ops_arg_dat(rho_new, 1, S1D_0, "real(8)", OPS_WRITE), &
          & ops_arg_dat(rhou_new, 1, S1D_0, "real(8)", OPS_WRITE), &
          & ops_arg_dat(rhoE_new, 1, S1D_0, "real(8)", OPS_WRITE), &
          & ops_arg_dat(rhoin, 1, S1D_0, "real(8)", OPS_WRITE), &
          & ops_arg_idx())

  call ops_exit( )

end program SHSGC