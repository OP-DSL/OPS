program lowdim
    use OPS_Fortran_Reference
    use OPS_Fortran_hdf5_Declarations
    use OPS_CONSTANTS

    use, intrinsic :: ISO_C_BINDING

    implicit none

    type(ops_block)   :: grid3D
    
    ! ops_dats
    type(ops_dat)     :: d_dat3D
    type(ops_dat)     :: d_dat2D_XY, d_dat2D_YZ, d_dat2D_XZ
    type(ops_dat)     :: d_dat1D_X, d_dat1D_Y, d_dat1D_Z

    ! ops stencils
    TYPE(ops_stencil) :: s3d_000
    TYPE(ops_stencil) :: s3d_000_strid3d_x, s3d_000_strid3d_y, s3d_000_strid3d_z
    TYPE(ops_stencil) :: s3d_000_strid3d_xy, s3d_000_strid3d_xz, s3d_000_strid3d_yz

    integer(kind=4) :: d_p(3)
    integer(kind=4) :: d_m(3)

    ! size for OPS
    integer(kind=4) :: d_size(3)

    ! base
    integer(kind=4) :: d_base(3) = [1,1,1]   !this is in fortran indexing - start from 1

    ! null array - for declaring ops dat
    real(kind=8), dimension(:), allocatable :: temp_real_null

    integer(kind=4) :: a3d_000(3) = [0,0,0]

    integer(kind=4) :: stride3d_x(3) = [1,0,0]
    integer(kind=4) :: stride3d_y(3) = [0,1,0]
    integer(kind=4) :: stride3d_z(3) = [0,0,1]

    integer(kind=4) :: stride3d_xy(3) = [1,1,0]
    integer(kind=4) :: stride3d_xz(3) = [1,0,1]
    integer(kind=4) :: stride3d_yz(3) = [0,1,1]

    ! profiling
    real(kind=c_double) :: startTime = 0
    real(kind=c_double) :: endTime = 0

    integer(kind=4) :: rangexyz(6)
    real(kind=8) :: val

    character(len=60) :: fname
    character(len=3) :: pnxhdf
    parameter(pnxhdf = '.h5')

    !-----------------------OPS Initialization------------------------
    call ops_init(2)

    !-----------------------OPS Declarations--------------------------

    ! declare block
    call ops_decl_block(3, grid3D, "block")

    ! declare dats
    d_size = [10,10,10]
    d_m    = [-1,-1,-1]
    d_p    = [1,1,1]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat3D, "real(kind=8)", "dat3D")

    d_size = [10,10,1]
    d_m    = [-1,-1,0]
    d_p    = [1,1,0]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat2D_XY, "real(kind=8)", "dat2D_XY")

    d_size = [1,10,10]
    d_m    = [0,-1,-1]
    d_p    = [0,1,1]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat2D_YZ, "real(kind=8)", "dat2D_YZ")

    d_size = [10,1,10]
    d_m    = [-1,0,-1]
    d_p    = [1,0,1]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat2D_XZ, "real(kind=8)", "dat2D_XZ")

    d_size = [10,1,1]
    d_m    = [-1,0,0]
    d_p    = [1,0,0]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat1D_X, "real(kind=8)", "dat1D_X")

    d_size = [1,10,1]
    d_m    = [0,-1,0]
    d_p    = [0,1,0]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat1D_Y, "real(kind=8)", "dat1D_Y")

    d_size = [1,1,10]
    d_m    = [0,0,-1]
    d_p    = [0,0,1]
    call ops_decl_dat(grid3D, 1, d_size, d_base, d_m, d_p, temp_real_null, d_dat1D_Z, "real(kind=8)", "dat1D_Z")

    ! declare stencils
    call ops_decl_stencil( 3, 1, a3d_000, s3d_000, "S3D_000")

    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_x, s3d_000_strid3d_x, "s2D_000_stride3D_x")
    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_y, s3d_000_strid3d_y, "s2D_000_stride3D_y")
    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_z, s3d_000_strid3d_z, "s2D_000_stride3D_z")

    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_xy, s3d_000_strid3d_xy, "s2D_000_stride3D_xy")
    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_yz, s3d_000_strid3d_yz, "s2D_000_stride3D_yz")
    call ops_decl_strided_stencil( 3, 1, a3d_000, stride3d_xz, s3d_000_strid3d_xz, "s2D_000_stride3D_xz")

    ! start timer
    call ops_timers ( startTime )

    call ops_partition("")

    val = 0.0_8
    rangexyz = [1,10,1,10,1,10]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat3D, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 1.0_8
    rangexyz = [1,10,1,10,1,1]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat2D_XY, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 2.0_8
    rangexyz = [1,1,1,10,1,10]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat2D_YZ, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 3.0_8
    rangexyz = [1,10,1,1,1,10]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat2D_XZ, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 4.0_8
    rangexyz = [1,10,1,1,1,1]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat1D_X, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 5.0_8
    rangexyz = [1,1,1,10,1,1]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat1D_Y, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    val = 6.0_8
    rangexyz = [1,1,1,1,1,10]
    call ops_par_loop(lowdim_kernel_set_val, "set value", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat1D_Z, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_gbl(val, 1, "real(kind=8)", OPS_READ))

    rangexyz = [1,10,1,10,1,10]
    call ops_par_loop(lowdim_kernel_calc, "calc", grid3D, 3, rangexyz,  &
                    ops_arg_dat(d_dat3D, 1, s3d_000, "real(kind=8)", OPS_WRITE), &
                    ops_arg_dat(d_dat2D_XY, 1, s3d_000_strid3d_xy, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_dat2D_YZ, 1, s3d_000_strid3d_yz, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_dat2D_XZ, 1, s3d_000_strid3d_xz, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_dat1D_X, 1, s3d_000_strid3d_x, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_dat1D_Y, 1, s3d_000_strid3d_y, "real(kind=8)", OPS_READ), &
                    ops_arg_dat(d_dat1D_Z, 1, s3d_000_strid3d_z, "real(kind=8)", OPS_READ))
 
    fname = 'output'//pnxhdf
    call ops_dump_to_hdf5(trim(fname))

    call ops_timers( endTime )

    IF (ops_is_root() == 1) THEN
        write(*,'(a)') 'This run is considered PASSED'
        write(*,'(a,f16.7,a)')  ' completed in ', endTime - startTime, ' seconds'
    END IF

    call ops_exit( )

end program lowdim
