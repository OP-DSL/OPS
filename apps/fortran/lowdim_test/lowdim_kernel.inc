SUBROUTINE lowdim_kernel_set_val(farray, val)

    real(kind=8), dimension(1) :: farray
    real(kind=8), intent(in) :: val

    farray(OPS_ACC1(0,0,0)) = val

END SUBROUTINE lowdim_kernel_set_val

SUBROUTINE lowdim_kernel_calc(dat3D,dat2D_xy,dat2D_yz,dat2D_xz,dat1D_x,dat1D_y,dat1D_z)

    real(kind=8), dimension(1) :: dat3D
    real(kind=8), dimension(1), intent(in) :: dat2D_xy,dat2D_yz,dat2D_xz
    real(kind=8), dimension(1), intent(in) :: dat1D_x,dat1D_y,dat1D_z

    dat3D(OPS_ACC1(0,0,0)) = dat2D_xy(OPS_ACC2(0,0,0)) + &
                             dat2D_yz(OPS_ACC3(0,0,0)) + &
                             dat2D_xz(OPS_ACC4(0,0,0)) + &
                             dat1D_x(OPS_ACC5(0,0,0))  + &
                             dat1D_y(OPS_ACC6(0,0,0))  + &
                             dat1D_z(OPS_ACC7(0,0,0))

END SUBROUTINE lowdim_kernel_calc
