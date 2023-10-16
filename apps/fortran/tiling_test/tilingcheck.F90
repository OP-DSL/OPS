program tilingcheck
    use OPS_Fortran_Reference
    use OPS_CONSTANTS

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_block)   :: grid1D

    integer(kind=4) :: d_size(1)
    integer(kind=4) :: d_base(1) = [1] !array indexing - start from 1
    integer(kind=4) :: d_p(1) !max boundary depths for the dat in the possitive direction
    integer(kind=4) :: d_m(1) !max boundary depths for the dat in the negative direction

    real(kind=8), dimension(:), allocatable :: temp_real_null
    integer(kind=4), dimension(:), allocatable :: temp_int_null

    integer(kind=4) :: ispec,iindex
    integer(kind=4) :: rangexyz(2)
    character(len=20) :: buf
    integer(kind=4) :: nxglbl = 64
    integer(kind=4) :: nhalox = 5

    integer(kind=4) :: a1d_0(1) = [0]
    integer(kind=4) :: a1d_p5_to_m5_x(11) = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
    
    TYPE(ops_dat) :: d_store1, d_store2, d_store3, d_store4, d_store5, d_store6, d_divm
    TYPE(ops_dat) :: d_ucor, d_vcor, d_wcor

    TYPE(ops_dat) :: d_wd1x, d_pd1x, d_td1x
    TYPE(ops_dat) :: d_wd2x, d_pd2x, d_td2x
    TYPE(ops_dat) :: d_ufxl, d_vfxl, d_wfxl
    TYPE(ops_dat) :: d_drun, d_urun, d_vrun, d_wrun, d_erun
    TYPE(ops_dat) :: d_drhs, d_urhs, d_vrhs, d_wrhs, d_erhs
    TYPE(ops_dat) :: d_derr, d_uerr, d_verr, d_werr, d_eerr

    TYPE(ops_dat) :: d_utmp, d_vtmp, d_wtmp, d_prun, d_trun, d_transp, d_store7

    TYPE(ops_dat) :: d_itndex(2)
    TYPE(ops_dat) :: d_yrhs(2), d_yrun(2), d_yerr(2), d_rate(2), d_rrte(2)

    TYPE(ops_stencil) :: s1d_0
    TYPE(ops_stencil) :: s1d_p5_to_m5_x

!   ---------------------- Declarations end here ------------------------------
    call ops_init(6)
    call ops_decl_block(1, grid1D, "SENGA_GRID")

    d_size = [nxglbl]
    d_m    = [0]
    d_p    = [0]

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store1, "real(kind=8)", "STORE1")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store2, "real(kind=8)", "STORE2")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store3, "real(kind=8)", "STORE3")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store4, "real(kind=8)", "STORE4")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store5, "real(kind=8)", "STORE5")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store6, "real(kind=8)", "STORE6")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_divm, "real(kind=8)", "DIVM")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_ucor, "real(kind=8)", "UCOR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_vcor, "real(kind=8)", "VCOR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wcor, "real(kind=8)", "WCOR")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wd1x, "real(kind=8)", "WD1X")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_pd1x, "real(kind=8)", "PD1X")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_td1x, "real(kind=8)", "TD1X")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wd2x, "real(kind=8)", "WD2X")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_pd2x, "real(kind=8)", "PD2X")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_td2x, "real(kind=8)", "TD2X")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_ufxl, "real(kind=8)", "UFXL")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_vfxl, "real(kind=8)", "VFXL")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wfxl, "real(kind=8)", "WFXL")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_drun, "real(kind=8)", "DRUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_urun, "real(kind=8)", "URUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_vrun, "real(kind=8)", "VRUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wrun, "real(kind=8)", "WRUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_erun, "real(kind=8)", "ERUN")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_derr, "real(kind=8)", "DERR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_uerr, "real(kind=8)", "UERR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_verr, "real(kind=8)", "VERR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_werr, "real(kind=8)", "WERR")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_eerr, "real(kind=8)", "EERR")

!---------------------------------------MULTI-DIM DAT--------------------------------------------------------

    d_size = [nxglbl]
    d_m    = [0]
    d_p    = [0]    
    
    DO ispec = 1,2
        write(buf,"(A4,I2)") "YRUN",ispec
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_yrun(ispec), "real(kind=8)", buf)
        write(buf,"(A4,I2)") "YERR",ispec
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_yerr(ispec), "real(kind=8)", buf)
        write(buf,"(A4,I2)") "RATE",ispec
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_rate(ispec), "real(kind=8)", buf)
        write(buf,"(A4,I2)") "RRTE",ispec
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_rrte(ispec), "real(kind=8)", buf)
    END DO


!--------------------------------------- with halos -----------------------------------------------------------

    d_size = [nxglbl]
    d_m    = [-nhalox]
    d_p    = [nhalox]
    
    DO iindex = 1,2
        write(buf,"(A6,I2)") "ITNDEX",iindex
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_int_null, d_itndex(iindex), "integer(kind=4)", buf)
    END DO

    DO ispec = 1,2
        write(buf,"(A4,I2)") "YRHS",ispec
        call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_yrhs(ispec), "real(kind=8)", buf)
    END DO


    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_drhs, "real(kind=8)", "DRHS")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_urhs, "real(kind=8)", "URHS")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_vrhs, "real(kind=8)", "VRHS")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wrhs, "real(kind=8)", "WRHS")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_erhs, "real(kind=8)", "ERHS")

    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_utmp, "real(kind=8)", "UTMP")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_vtmp, "real(kind=8)", "VTMP")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_wtmp, "real(kind=8)", "WTMP")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_prun, "real(kind=8)", "PRUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_trun, "real(kind=8)", "TRUN")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_transp, "real(kind=8)", "TRANSP")
    call ops_decl_dat(grid1D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_store7, "real(kind=8)", "STORE7")

!-------------------------------------------------------------------------------------
    call ops_decl_stencil( 1, 1, a1d_0, s1d_0, "0")
    call ops_decl_stencil( 1, 11, a1d_p5_to_m5_x, s1d_p5_to_m5_x, "5 to -5")


!-------------------------------------------------------------------------------------
    call ops_partition(" ")

!-------------------------------------------------------------------------------------

    rangexyz = [1,nxglbl]
    DO ispec = 1,2
       call ops_par_loop(user_kernel_eqA, "eqA rhscal 625", grid1D, 1, rangexyz, &
                     ops_arg_dat(d_rrte(ispec), 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                     ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_READ))
    END DO

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqB, "eqB rhscal 713", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_WRITE))

    rangexyz = [1-nhalox,nxglbl+nhalox]
    call ops_par_loop(user_kernel_eqB, "eqB rhscal 721", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_wtmp, 1, s1d_0, "real(kind=8)", OPS_WRITE))    

    DO ispec = 1,2
        rangexyz = [1-nhalox,nxglbl+nhalox]
        call ops_par_loop(user_kernel_eqC, "eqC rhscal 816", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_0, "real(kind=8)", OPS_RW), &
                        ops_arg_dat(d_drhs, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqD, "eqD rhscal 828", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_divm, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1-nhalox,nxglbl+nhalox]
        call ops_par_loop(user_kernel_eqE, "eqE rhscal 845", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 850", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_store7, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_WRITE))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqG, "eqG rgscal 874", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 897", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_WRITE))


        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqH, "eqH rhscal 983", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_vrhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_wrhs, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1-nhalox,nxglbl+nhalox]
        call ops_par_loop(user_kernel_eqA, "eqA rhscal 1013", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                        ops_arg_dat(d_transp, 1, s1d_0, "real(kind=8)", OPS_READ))

        
        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqD, "eqD rhscal 1116", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_ucor, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ))

        call ops_par_loop(user_kernel_eqD, "eqD rhscal 1121", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_vcor, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ))

        call ops_par_loop(user_kernel_eqD, "eqD rhscal 1126", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_wcor, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1-nhalox,nxglbl+nhalox]
        iindex = 1 + (ispec-1)/2
        call ops_par_loop(user_kernel_eqI, "eqI rhscal 1151", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_utmp, 1, s1d_0, "real(kind=8)", OPS_RW), &
                        ops_arg_dat(d_wtmp, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_trun, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_itndex(iindex), 1, s1d_0, "integer(kind=4)", OPS_READ), &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 1254", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_store7, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_WRITE))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqJ, "eqJ rhscal 1270", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store5, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store6, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqK, "eqK rhscal 1295", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_erhs, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store5, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store6, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_utmp, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 1318", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_utmp, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_WRITE))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqK, "eqK rhscal 1349", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_erhs, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store5, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store6, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqA, "eqA rhscal 1455", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                        ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 1468", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_WRITE))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqL, "eqL rhscal 1487", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqM, "eqM rhscal 1510", grid1D, 1, rangexyz,  &
                        ops_arg_dat(d_erhs, 1, s1d_0, "real(kind=8)", OPS_INC), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_utmp, 1, s1d_0, "real(kind=8)", OPS_READ))

    END DO

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 2553", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_wtmp, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_WRITE))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqA, "eqA rhscal 2585", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_READ))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqN, "eqN rhscal 2604", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_erhs, 1, s1d_0, "real(kind=8)", OPS_INC), &
                    ops_arg_dat(d_wtmp, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_ucor, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_vcor, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_wcor, 1, s1d_0, "real(kind=8)", OPS_READ))

    DO ispec = 1,2
        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqF, "eqF rhscal dfbydx 2693", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_WRITE))

        rangexyz = [1,nxglbl]
        call ops_par_loop(user_kernel_eqO, "eqO rhscal 2713", grid1D, 1, rangexyz, &
                        ops_arg_dat(d_yrhs(ispec), 1, s1d_0, "real(kind=8)", OPS_RW), &
                        ops_arg_dat(d_rate(ispec), 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_ucor, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store2, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_vcor, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_store3, 1, s1d_0, "real(kind=8)", OPS_READ), &
                        ops_arg_dat(d_wcor, 1, s1d_0, "real(kind=8)", OPS_READ))
    END DO

!--------------------------------------- RHSVEL ----------------------------------------

    rangexyz = [1-nhalox,nxglbl+nhalox]
    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 58", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_utmp, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_drhs, 1, s1d_0, "real(kind=8)", OPS_READ))
    
    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 63", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_vrhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_drhs, 1, s1d_0, "real(kind=8)", OPS_READ))

    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 68", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_wtmp, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_wrhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_drhs, 1, s1d_0, "real(kind=8)", OPS_READ))

    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 159", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_utmp, 1, s1d_0, "real(kind=8)", OPS_READ))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqF, "eqF rhsvel dfbydx 166", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_store7, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_WRITE))

    rangexyz = [1-nhalox,nxglbl+nhalox]
    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 179", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_vtmp, 1, s1d_0, "real(kind=8)", OPS_READ))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqF, "eqF rhsvel dfbydx 187", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_store7, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store5, 1, s1d_0, "real(kind=8)", OPS_WRITE))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqP, "eqP rhsvel 196", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_store4, 1, s1d_0, "real(kind=8)", OPS_INC), &
                    ops_arg_dat(d_store1, 1, s1d_0, "real(kind=8)", OPS_READ))

    rangexyz = [1-nhalox,nxglbl+nhalox]
    call ops_par_loop(user_kernel_eqE, "eqE rhsvel 207", grid1D, 1, rangexyz,  &
                    ops_arg_dat(d_store7, 1, s1d_0, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_urhs, 1, s1d_0, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_wtmp, 1, s1d_0, "real(kind=8)", OPS_READ))

    rangexyz = [1,nxglbl]
    call ops_par_loop(user_kernel_eqF, "eqF rhsvel dfbydx 215", grid1D, 1, rangexyz, &
                    ops_arg_dat(d_store7, 1, s1d_p5_to_m5_x, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_store6, 1, s1d_0, "real(kind=8)", OPS_WRITE))

#ifdef OPS_LAZY
    call ops_execute()
#endif

    call ops_exit()

end program tilingcheck
