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
    use OPS_Fortran_Declarations
    use OPS_Fortran_RT_Support

    use, intrinsic :: ISO_C_BINDING

    implicit none

    ! size along x
    integer, parameter :: imax=4094
    ! size along y
    integer, parameter :: jmax=4094
    ! max iterations
    integer, parameter :: iter_max=100
    integer :: i, j, iter

    real(8), dimension (:), allocatable :: A, Anew

    real(8), parameter :: pi=2.0_8*asin(1.0_8)
    real(8), parameter :: tol=1.0e-6_8
    real(8) :: err_diff

    ! integer references (valid inside the OPS library) for ops_block
    type(ops_block)   :: grid2D

    ! ops_dats
    type(ops_dat)     ::    d_A, d_Anew

    ! vars for stencils
    integer s2D_00(2) /0,0/
    type(ops_stencil) :: S2D_0pt

    integer s2D_05(10) /0,0, 1,0, -1,0, 0,1, 0,-1/
    type(ops_stencil) :: S2D_5pt

    !vars for reduction
    real(8) :: error

    integer d_p(2) /1,1/   !max boundary depths for the dat in the possitive direction
    integer d_m(2) /-1,-1/ !max boundary depths for the dat in the negative direction

    ! size for OPS
    integer size(2)

    ! base index
    integer base(2) /0,0/

    ! profiling
    real(kind=c_double) :: startTime = 0
    real(kind=c_double) :: endTime = 0

    allocate (A(0:((jmax+2)*(imax+2))-1), Anew(0:((jmax+2)*(imax+2))-1))
    ! Initialize
    A = 0.0_8

    size(1) = jmax
    size(2) = imax

    !-----------------------OPS Initialization------------------------
    call ops_init(2)

    !-----------------------OPS Declarations--------------------------

    !declare block
    call ops_decl_block(2, grid2D, "grid2D")

    !declare stencils
    call ops_decl_stencil( 2, 1, s2D_00, S2D_0pt, "0pt_stencil")
    call ops_decl_stencil( 2, 5, s2D_05, S2D_5pt, "5pt_stencil")

    !declare data on blocks

    !declare ops_dat
    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, A, d_A, "real(8)", "A")
    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, Anew, d_Anew, "real(8)", "Anew")

    error=1.0_8 
    
    ! start timer
    call ops_timers ( startTime )
 
    ! Bottom
    do i=0,imax+1
        A((0)*(imax+2)+i)   = 0.0_8
    end do

    ! Top
    do i=0,imax+1
        A((jmax+1)*(imax+2)+i) = 0.0_8
    end do

    ! Left
    do j=0,jmax+1
        A((j)*(imax+2)+0)   = sin(pi * j/(jmax+1))
    end do

    ! Right
    do j=0,jmax+1
        A((j)*(imax+2)+imax+1) = sin(pi * j/(jmax+1)) * exp(-pi)
    end do

    write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', imax+2, ' x', jmax+2, ' mesh'

    iter=0

    do i=1,imax+1
        Anew((0)*(imax+2)+i)   = 0.0_8
    end do

    do i=1,imax+1
        Anew((jmax+1)*(imax+2)+i) = 0.0_8
    end do

    do j=1,jmax+1
        Anew((j)*(imax+2)+0)   = sin(pi * j/(jmax+1))
    end do

    do j=1,jmax+1
        Anew((j)*(imax+2)+imax+1) = sin(pi * j/(jmax+1)) * exp(-pi)
    end do

    do while ( error .gt. tol .and. iter .lt. iter_max )
        error=0.0_8

        do i=1,imax
            do j=1,jmax
                Anew((j)*(imax+2)+i) = 0.25_8 * ( A((j  )*(imax+2)+ i+1) + A((j  )*(imax+2)+ i-1) &
                                             &  + A((j-1)*(imax+2)+ i  ) + A((j+1)*(imax+2)+ i  ) )
                error = max( error, abs( Anew((j)*(imax+2)+i)-A((j)*(imax+2)+i) ) )
            end do
        end do

        do i=1,imax
            do j=1,jmax
                A((j)*(imax+2)+i) = Anew((j)*(imax+2)+i)
            end do
        end do

        if(mod(iter,10).eq.0 ) write(*,'(i5,a,f16.7)') iter, ', ',error
            iter = iter +1

    end do  ! End of do while loop

    write(*,'(i5,a,f16.7)') iter, ', ',error

    err_diff = abs((100.0*(error/2.421354960840227e-03))-100.0)

    write(*,'(a,e18.5,a)') 'Total error is within ', err_diff,' % of the expected error'

    if(err_diff .lt. 0.001_8) then
        write(*,'(a)') 'This run is considered PASSED'
    else
        write(*,'(a)') 'This test is considered FAILED'
    end if    

    call ops_timers( endTime )
    write(*,'(a,f16.7,a)')  ' completed in ', endTime - startTime, ' seconds'

    call ops_exit( )

    deallocate (A,Anew)

end program laplace
