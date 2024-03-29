! @brief SHSGC top level program
! @author Satya P. Jammy, converted to OPS by Gihan Mudalige
! @details
!
! This version is based on Fortran and uses the OPS prototype highlevel domain
! specific API for developing multi-block Structured mesh applications


program SHSGC
  use OPS_Fortran_Reference
  use OPS_FORTRAN_HDF5_DECLARATIONS
  use OPS_CONSTANTS

  use, intrinsic :: ISO_C_BINDING

  implicit none

  intrinsic :: sqrt, real

  integer(kind=4) ::  niter, iter, nrk
  real(8) :: totaltime
  real(8) :: local_rms

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

  !ops_reduction
  type(ops_reduction) :: rms

  ! vars for stencils
  integer(kind=4) ::  S1D_0_array(1) = [0]
  integer(kind=4) ::  S1D_01_array(2) = [0,1]
  integer(kind=4) ::  S1D_0M1_array(2) = [0,-1]
  integer(kind=4) ::  S1D_0M1M2P1P2_array(5) = [0,-1,-2,1,2]
  type(ops_stencil) :: S1D_0, S1D_01, S1D_0M1
  type(ops_stencil) :: S1D_0M1M2P1P2

  ! vars for halo_depths
  integer(kind=4) ::  d_p(1) = [2]   !max halo depths for the dat in the possitive direction
  integer(kind=4) ::  d_m(1) = [-2] !max halo depths for the dat in the negative direction

  !base
  integer(kind=4) ::  base(1) = [1] ! this is in fortran indexing

  !size
  integer(kind=4) ::  size(1) = [204] !size of the dat -- should be identical to the block on which its define on

  !null array
  real(kind=c_double), dimension(:), allocatable :: temp
  ! pointer for ops_fetch_dat()
  real(kind=c_double), dimension(:), allocatable :: u_rho_new

  !iteration range
  !iterange needs to be fortran indexed here
  ! inclusive indexing for both min and max points in the range
  !.. but internally will convert to c index
  integer(kind=4) ::  nxp_range(2), nxp_range_1(2), nxp_range_2(2), nxp_range_3(2), &
  & nxp_range_4(2), nxp_range_5(2)

  ! profiling
  real(kind=c_double) :: startTime = 0
  real(kind=c_double) :: endTime = 0

  !rk3 constants
  real(8) :: a1(3)
  real(8) :: a2(3)

  !status variable to check success of ops_fetch_dat()
  integer(kind=4) :: status

  !for validation
  real(8) :: validate_rms, rms_diff

  character(len=60) :: fname
  character(len=3) :: pnxhdf
  parameter(pnxhdf = '.h5')

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
  dt=0.0002_8
  del2 = 1e-8_8
  akap2 = 0.40_8
  tvdsmu = 0.25_8
  con = tvdsmu**2.0_8

  totaltime = 0.0_8
  !Initialize rk3 co-efficients
  a1(1) = 2.0_8/3.0_8
  a1(2) = 5.0_8/12.0_8
  a1(3) = 3.0_8/5.0_8
  a2(1) = 1.0_8/4.0_8
  a2(2) = 3.0_8/20.0_8
  a2(3) = 3.0_8/5.0_8

  !-------------------------- Initialisation --------------------------

  ! OPS initialisation
  call ops_init(1)
  call ops_set_soa(1)

  !----------------------------OPS Declarations------------------------

  ! declare block
  call ops_decl_block(1, shsgc_grid, "shsgc grid")

  ! declare stencils
  call ops_decl_stencil( 1, 1, S1D_0_array, S1D_0, "0")
  call ops_decl_stencil( 1, 2, S1D_01_array, S1D_01, "0,1")
  call ops_decl_stencil( 1, 2, S1D_0M1_array, S1D_0M1, "0,-1")
  call ops_decl_stencil( 1, 5, S1D_0M1M2P1P2_array, S1D_0M1M2P1P2, "0,-1,-2,1,2")


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

  ! extra dat for rhoin
  call ops_decl_dat(shsgc_grid, 1, size, base, d_m, d_p, temp, rhoin, "real(8)", "rhoin")

  ! TVD scheme variables
  call ops_decl_dat(shsgc_grid, 9, size, base, d_m, d_p, temp, r, "real(8)", "r")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, al, "real(8)", "al")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, alam, "real(8)", "alam")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, gt, "real(8)", "gt")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, tht, "real(8)", "tht")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, ep2, "real(8)", "ep2")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, cmp, "real(8)", "cmp")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, cf, "real(8)", "cf")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, eff, "real(8)", "eff")
  call ops_decl_dat(shsgc_grid, 3, size, base, d_m, d_p, temp, s, "real(8)", "s")

  ! reduction handle for rms variable
  call ops_decl_reduction_handle(8, rms, "real(8)", "rms")

  call ops_partition("1D_BLOCK_DECOMPOSE")

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


      !call ops_print_dat_to_txtfile(rho_new, "shsgc.dat")
      !call exit()

    !if (iter .eq. 2) then
    !call ops_print_dat_to_txtfile(rho_new, "shsgc.dat")
    !call exit()
    !end if

  ! start timer
  call ops_timers(startTime)

  !
  ! main iterative loop
  !
  niter = 9005
  DO iter = 1, niter

    !Save previous data arguments
    call ops_par_loop(save_kernel, "save_kernel", shsgc_grid, 1, nxp_range, &
            & ops_arg_dat(rho_old, 1, S1D_0, "real(8)", OPS_WRITE), &
            & ops_arg_dat(rhou_old, 1, S1D_0, "real(8)", OPS_WRITE), &
            & ops_arg_dat(rhoE_old, 1, S1D_0, "real(8)", OPS_WRITE), &
            & ops_arg_dat(rho_new, 1, S1D_0, "real(8)", OPS_READ), &
            & ops_arg_dat(rhou_new, 1, S1D_0, "real(8)", OPS_READ), &
            & ops_arg_dat(rhoE_new, 1, S1D_0, "real(8)", OPS_READ))

    !rk3 loop
    DO nrk = 1, 3

      ! make residue equal to zero
      call ops_par_loop(zerores_kernel, "zerores_kernel", shsgc_grid, 1, nxp_range, &
            & ops_arg_dat(rho_res, 1, S1D_0, "real(8)", OPS_WRITE), &
            & ops_arg_dat(rhou_res, 1, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(rhoE_res, 1, S1D_0, "real(8)",OPS_WRITE))

      ! computations of convective derivatives
      ! TODO

      ! calculate drhou/dx
      nxp_range_1(1) = 3
      nxp_range_1(2) = nxp-2
      call ops_par_loop(drhoudx_kernel, "drhoudx_kernel", shsgc_grid, 1, nxp_range_1, &
            & ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rho_res, 1, S1D_0, "real(8)",OPS_WRITE))

      ! calculate d(rhouu + p)/dx

      call ops_par_loop(drhouupdx_kernel, "drhouupdx_kernel", shsgc_grid, 1, nxp_range_1, &
            & ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rho_new,  1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rhoE_new, 1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rhou_res, 1, S1D_0, "real(8)",OPS_WRITE))

      !if (iter .eq. 6001) then
      !call ops_print_dat_to_txtfile(rhou_res, "shsgc.dat")
      !call exit()
      !end if

      ! Energy equation derivative d(rhoE+p)u/dx
      call ops_par_loop(drhoEpudx_kernel, "drhoEpudx_kernel", shsgc_grid, 1, nxp_range_1, &
            & ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rho_new,  1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rhoE_new, 1, S1D_0M1M2P1P2, "real(8)",OPS_READ), &
            & ops_arg_dat(rhoE_res, 1, S1D_0, "real(8)",OPS_WRITE))

      ! update use rk3 co-efficients
      nxp_range_2(1) = 4
      nxp_range_2(2) = nxp-2
      call ops_par_loop(updateRK3_kernel, "updateRK3_kernel", shsgc_grid, 1, nxp_range_2, &
            & ops_arg_dat(rho_new,  1, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(rhou_new, 1, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(rhoE_new, 1, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(rho_old,  1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(rhou_old, 1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(rhoE_old, 1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(rho_res,  1, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(rhou_res, 1, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(rhoE_res, 1, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_gbl(a1(nrk), 1, "real(8)", OPS_READ), &
            & ops_arg_gbl(a2(nrk), 1, "real(8)", OPS_READ))

    END DO

    !
    ! TVD scheme
    !



    ! Riemann invariants
    nxp_range_3(1) = 1
    nxp_range_3(2) = nxp-1
    call ops_par_loop(Riemann_kernel, "Riemann_kernel", shsgc_grid, 1, nxp_range_3, &
            & ops_arg_dat(rho_new,  1, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(rhou_new,  1, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(rhoE_new,  1, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(alam,  3, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(r,  9, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(al, 3, S1D_0, "real(8)",OPS_WRITE))

    ! limiter function
    nxp_range_4(1) = 2
    nxp_range_4(2) = nxp-1
    call ops_par_loop(limiter_kernel, "limiter_kernel", shsgc_grid, 1, nxp_range_4, &
            & ops_arg_dat(al, 3, S1D_0M1, "real(8)",OPS_READ), &
            & ops_arg_dat(tht,3, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(gt, 3, S1D_0, "real(8)",OPS_WRITE))

    ! Second order tvd dissipation
    call ops_par_loop(tvd_kernel, "tvd_kernel", shsgc_grid, 1, nxp_range_3, &
            & ops_arg_dat(tht, 3, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(ep2, 3, S1D_0, "real(8)",OPS_WRITE))

    ! vars
    call ops_par_loop(vars_kernel, "vars_kernel", shsgc_grid, 1, nxp_range_3, &
            & ops_arg_dat(alam,3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(al,  3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(gt,  3, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(cmp, 3, S1D_0, "real(8)",OPS_WRITE), &
            & ops_arg_dat(cf,  3, S1D_0, "real(8)",OPS_WRITE))

    ! cal upwind eff
    call ops_par_loop(calupwindeff_kernel, "calupwindeff_kernel", shsgc_grid, 1, nxp_range_3, &
            & ops_arg_dat(cmp,3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(gt, 3, S1D_01, "real(8)",OPS_READ), &
            & ops_arg_dat(cf, 3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(al, 3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(ep2,3, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(r,  9, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_dat(eff,3, S1D_0, "real(8)",OPS_WRITE))

    ! fact
    call ops_par_loop(fact_kernel, "fact_kernel", shsgc_grid, 1, nxp_range_4, &
            & ops_arg_dat(eff,  3, S1D_0M1, "real(8)",OPS_READ), &
            & ops_arg_dat(s,    3, S1D_0,   "real(8)",OPS_WRITE))

    ! update loop
    nxp_range_5(1) = 4
    nxp_range_5(2) = nxp-3
    call ops_par_loop(update_kernel, "update_kernel", shsgc_grid, 1, nxp_range_5, &
            & ops_arg_dat(rho_new,  1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(rhou_new, 1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(rhoE_new, 1, S1D_0, "real(8)",OPS_RW), &
            & ops_arg_dat(s,        3, S1D_0, "real(8)",OPS_READ))

    totaltime = totaltime + dt
    if (ops_is_root() .eq. 1) then
      write (*,*) iter, totaltime
    endif

#ifdef OPS_LAZY
    call ops_execute()
#endif

  ENDDO

  call ops_timers(endTime)

  ! compare solution to referance solution
  local_rms = 0.0_8
  call ops_par_loop(test_kernel, "test_kernel", shsgc_grid, 1, nxp_range, &
            & ops_arg_dat(rho_new,  1, S1D_0, "real(8)",OPS_READ), &
            & ops_arg_reduce(rms, 1, "real(8)", OPS_INC))


  call ops_reduction_result(rms, local_rms)

  if (ops_is_root() .eq. 1) then
    write (*,*) 'Max total runtime =', endTime - startTime,'seconds'

    validate_rms = sqrt(local_rms)/nxp
    rms_diff=ABS((100.0_8*(validate_rms/0.233688543536201_8))-100.0_8)
    write (*,'(a,f16.7)') "RMS = ", validate_rms !Correct RMS = 0.233689
    write (*,'(a,e16.7,a)') "Total error is within",rms_diff,"% of the expected error"

    IF(rms_diff.LT.0.001) THEN
      write(*,'(a)')"This test is considered PASSED"
    ELSE
      write(*,'(a)')"This test is considered FAILED"
    ENDIF

  end if

!  call ops_print_dat_to_txtfile(rho_new, "shsgc.dat")

  fname = 'shsgc'//pnxhdf
  call ops_fetch_block_hdf5_file(shsgc_grid, trim(fname))
  call ops_fetch_dat_hdf5_file(rho_new, trim(fname))

  call ops_fetch_dat(rho_new, u_rho_new, status);
  if (status .lt. 0)then
    write (*,*) 'ops_fetch_dat falied with status:', status
  end if

  call ops_exit( )

end program SHSGC
