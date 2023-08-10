PROGRAM mathtest
    use OPS_Fortran_Reference
    use OPS_Fortran_hdf5_Declarations
    use OPS_CONSTANTS

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_block)   :: grid3D

    type(ops_dat)     ::    d_sin, d_cos, d_log, d_exp

    integer s3D_00(3) /0,0,0/

    type(ops_stencil) :: S3D_0pt

    integer n
    integer d_p(3) /1,1,1/
    integer d_m(3) /-1,-1,-1/

    integer size(3)
    integer range(6)

    integer base(3) /1,1,1/

    real(8), dimension(:), allocatable :: temp

    CHARACTER (LEN=60) :: fname
    CHARACTER (LEN=3) :: pnxres
    PARAMETER(pnxres = '.h5')

    size = (/4,4,4/)
    
    call ops_init(2)

    call ops_decl_block(3, grid3D, "grid3D")


    call ops_decl_stencil( 3, 1, s3D_00, S3D_0pt, "0pt_stencil")

    call ops_decl_dat(grid3D, 1, size, base, d_m, d_p, temp, d_sin, "real(8)", "SIN")
    call ops_decl_dat(grid3D, 1, size, base, d_m, d_p, temp, d_cos, "real(8)", "COS")
    call ops_decl_dat(grid3D, 1, size, base, d_m, d_p, temp, d_log, "real(8)", "LOG")
    call ops_decl_dat(grid3D, 1, size, base, d_m, d_p, temp, d_exp, "real(8)", "EXP")

    range = (/1,4,1,4,1,4/)
    call ops_par_loop(math_kernel_init, "init", grid3D, 3, range, &
                    ops_arg_dat(d_sin, 1, S3D_0pt, "real(8)", OPS_WRITE), &
                    ops_arg_dat(d_cos, 1, S3D_0pt, "real(8)", OPS_WRITE), &
                    ops_arg_dat(d_log, 1, S3D_0pt, "real(8)", OPS_WRITE), &
                    ops_arg_dat(d_exp, 1, S3D_0pt, "real(8)", OPS_WRITE), &
                    ops_arg_idx())
                    
    
    fname = 'input_dats'//pnxres
    call ops_fetch_block_hdf5_file(grid3D, trim(fname))
    call ops_fetch_dat_hdf5_file(d_sin, trim(fname))
    call ops_fetch_dat_hdf5_file(d_cos, trim(fname))
    call ops_fetch_dat_hdf5_file(d_log, trim(fname))
    call ops_fetch_dat_hdf5_file(d_exp, trim(fname))

    DO n = 1, 2100
        call ops_par_loop(math_kernel_calc, "calc", grid3D, 3, range, &
                    ops_arg_dat(d_sin, 1, S3D_0pt, "real(8)", OPS_RW), &
                    ops_arg_dat(d_cos, 1, S3D_0pt, "real(8)", OPS_RW), &
                    ops_arg_dat(d_log, 1, S3D_0pt, "real(8)", OPS_RW), &
                    ops_arg_dat(d_exp, 1, S3D_0pt, "real(8)", OPS_RW))
    
    END DO

    fname = 'populated_dats'//pnxres
    call ops_fetch_block_hdf5_file(grid3D, trim(fname))
    call ops_fetch_dat_hdf5_file(d_sin, trim(fname))
    call ops_fetch_dat_hdf5_file(d_cos, trim(fname))
    call ops_fetch_dat_hdf5_file(d_log, trim(fname))
    call ops_fetch_dat_hdf5_file(d_exp, trim(fname))


    call ops_exit( )

END PROGRAM
