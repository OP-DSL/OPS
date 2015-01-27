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
  integer size(1) /204/ !size of the dat -- should be identical to the block on which its define on

  !null array
  real(kind=c_double) temp[allocatable](:)

  !iteration range
  !iterange needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index
  integer nxp_range(2)

  !-------------------------- Initialis constants--------------------------
  nxp = 204
  nyp = 5
  xhalo = 2
  yhalo = 2
  xmin = -5.0_8
  ymin = 0_8
  xmax = 5.0_8
  ymax = 0.5_8
  dx = (xmax-xmin)/(nxp-(1.0_8 + 2.0_8*xhalo))
  dy = (ymax-ymin)/(nyp-1.0_8)
  pl = 10.333_8
  pr = 1.0_8
  rhol = 3.857143_8
  rhor = 1.0_8
  ul = 2.6293690_8
  ur = 0.0_8
  gam = 1.4_8
  gam1=gam - 1.0_8
  eps = 0.2_8
  lambda = 5.0_8
  !a1[3]
  !a2[3]
  dt=0.0002_8
  del2 = 1e-8_8
  akap2 = 0.40_8
  tvdsmu = 0.25_8
  con = tvdsmu**2.0_8

  !-------------------------- Initialisation --------------------------

  !OPS initialisation
  call ops_init(2)

  !----------------------------OPS Declarations------------------------

  !declare block
  call ops_decl_block(1, shsgc_grid, "shsgc grid")

  !declare stencils
  call ops_decl_stencil( 2, 1, S1D_0_array, S1D_0, "0")
  call ops_decl_stencil( 2, 1, S1D_01_array, S1D_01, "0,1")
  call ops_decl_stencil( 2, 1, S1D_0M1_array, S1D_0M1, "0,-1")
  call ops_decl_stencil( 2, 1, S1D_0M1M2P1P2_array, S1D_0M1M2P1P2, "0,-1,-2,1,2")


  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, x, "double", "x")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_old, "double", "rho_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_new, "double", "rho_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rho_res, "double", "rho_res")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_old, "double", "rhou_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_new, "double", "rhou_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhou_res, "double", "rhou_res")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhov_old, "double", "rhov_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhov_new, "double", "rhov_new")

  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_old, "double", "rhoE_old")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_new, "double", "rhoE_new")
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoE_res, "double", "rhoE_res")

  !extra dat for rhoin
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoin, "double", "rhoin");


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

  call ops_print_dat_to_txtfile(rhoin, "shsgc.dat");

  call ops_exit( )

end program SHSGC