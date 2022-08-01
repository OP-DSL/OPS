!
!               ^
!               |        top
!               |    000000000000   r
!               | l  000000000000   i
!   y-direction | e  000000000000   g
!    (index-j)  | f  000000000000   h
!               | t  000000000000   t
!               |       bottom
!               o------------------>
!                      x-direction
!                       (index-i)

program laplace
    use OPS_Fortran_Reference
    use OPS_FORTRAN_RT_SUPPORT
    use OPS_CONSTANTS

    use, intrinsic :: ISO_C_BINDING

    implicit none

    ! max iterations
    integer, parameter :: iter_max=100
    integer :: i, j, iter

    real(8), parameter :: tol=1.0e-6_8
    real(8) :: err_diff

    ! integer references (valid inside the OPS library) for ops_block
    type(ops_block)   :: grid2D
    
    !ops_dats
    type(ops_dat)     ::    d_A, d_Anew
    
    !declare OPS constants
    call ops_decl_const("imax", 1, "int", imax)
    call ops_decl_const("jmax", 1, "int", jmax)
    call ops_decl_const("pi", 1, "double", pi)

    ! vars for stencils
    integer s2D_00(2) /0,0/
    type(ops_stencil) :: S2D_0pt
    
    integer s2D_05(10) /0,0, 1,0, -1,0, 0,1, 0,-1/
    type(ops_stencil) :: S2D_5pt

    !vars for reduction
    real(8) :: error
    type(ops_reduction) :: h_err

    integer d_p(2) /1,1/   !max boundary depths for the dat in the possitive direction
    integer d_m(2) /-1,-1/ !max boundary depths for the dat in the negative direction
    
    !size for OPS
    integer size(2)

    !base
    integer base(2) /1,1/   !this is in fortran indexing - start from 1

    !null array - for declaring ops dat    
    real(8), dimension(:), allocatable :: temp

    ! profiling
    real(kind=c_double) :: startTime = 0
    real(kind=c_double) :: endTime = 0

    ! iteration range - needs to be fortran indexed here
    ! inclusive indexing for both min and max points in the range
    !.. but internally will convert to c index

    integer :: bottom_range(4),top_range(4),left_range(4),right_range(4),interior_range(4)

    !initialize and declare constants
    imax = 4094
    jmax = 4094
    pi = 2.0_8*asin(1.0_8)

#ifdef OPS_WITH_CUDAFOR
    imax_OPS = imax
    jmax_OPS = jmax
    pi_OPS = pi
#endif
    
    size(1) = jmax
    size(2) = imax       

    !                         x(min,max)  y(min,max)  
    bottom_range   = [0,imax+1,      0,0]
    top_range      = [0,imax+1,      jmax+1,jmax+1]
    left_range     = [0,0,           0,jmax+1]
    right_range    = [imax+1,imax+1, 0,jmax+1]
    interior_range = [1,imax,      1,jmax]     

    !-----------------------OPS Initialization------------------------
    call ops_init(2)
    
    !-----------------------OPS Declarations--------------------------

    !declare block
    call ops_decl_block(2, grid2D, "grid2D")

    !declare stencils
    call ops_decl_stencil( 2, 1, s2D_00, S2D_0pt, "0pt_stencil");
    call ops_decl_stencil( 2, 5, s2D_05, S2D_5pt, "5pt_stencil");

    !declare data on blocks
    
    !declare ops_dat
    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_A, "real(8)", "A")
    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_Anew, "real(8)", "Anew")

    !declare reduction handles
    error=1.0_8 
    call ops_decl_reduction_handle(8, h_err, "real(8)", "error");
    
    ! start timer
    call ops_timers ( startTime )
    
    call ops_partition("")

    call ops_par_loop(set_zero_kernel, "set zero", grid2D, 2, bottom_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(8)", OPS_WRITE))
    
    call ops_par_loop(set_zero_kernel, "set zero", grid2D, 2, top_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(8)", OPS_WRITE))

    call ops_par_loop(left_bndcon_kernel, "left_bndcon", grid2D, 2, left_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(8)", OPS_WRITE), &
                    & ops_arg_idx());

    call ops_par_loop(right_bndcon_kernel, "right_bndcon", grid2D, 2, right_range, &
               & ops_arg_dat(d_A, 1, S2D_0pt, "real(8)", OPS_WRITE), &
               & ops_arg_idx());

    write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', imax+2, ' x', jmax+2, ' mesh'

    iter=0

    call ops_par_loop(set_zero_kernel, "set zero", grid2D, 2, bottom_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_WRITE))

    call ops_par_loop(set_zero_kernel, "set zero", grid2D, 2, top_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_WRITE))

    call ops_par_loop(left_bndcon_kernel, "left_bndcon", grid2D, 2, left_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_WRITE), &
                    & ops_arg_idx());

    call ops_par_loop(right_bndcon_kernel, "right_bndcon", grid2D, 2, right_range, &
               & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_WRITE), &
               & ops_arg_idx());

    do while ( error .gt. tol .and. iter .lt. iter_max )
        error=0.0_8

        call ops_par_loop(apply_stencil_kernel, "apply_stencil", grid2D, 2, interior_range, &
                        & ops_arg_dat(d_A,    1, S2D_5pt, "real(8)", OPS_READ), &
                        & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_WRITE), &
                        & ops_arg_reduce(h_err, 1, "real(8)", OPS_MAX));

        call ops_reduction_result(h_err, error);

        call ops_par_loop(copy_kernel, "copy", grid2D, 2, interior_range, &
                        & ops_arg_dat(d_A,    1, S2D_0pt, "real(8)", OPS_WRITE), &
                        & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(8)", OPS_READ));

        if(mod(iter,10).eq.0 ) write(*,'(i5,a,f16.7)') iter, ', ',error
            iter = iter +1

    end do  ! End of do while loop

    write(*,'(i5,a,f16.7)') iter, ', ',error

    err_diff = abs((100.0*(error/2.421354960840227e-03))-100.0);

    write(*,'(a,e18.5,a)') 'Total error is within ', err_diff,' % of the expected error'

    if(err_diff .lt. 0.001_8) then
        write(*,'(a)') 'This run is considered PASSED'
    else
        write(*,'(a)') 'This test is considered FAILED'
    end if    

    call ops_timers( endTime )
    write(*,'(a,f16.7,a)')  ' completed in ', endTime - startTime, ' seconds'

    call ops_exit( )

end program laplace
