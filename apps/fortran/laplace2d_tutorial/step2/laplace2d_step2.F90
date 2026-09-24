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

    real(8), dimension (:,:), allocatable :: A, Anew

    real(8), parameter :: pi=2.0_8*asin(1.0_8)
    real(8), parameter :: tol=1.0e-6_8
    real(8) :: error=1.0_8
    real(8) :: err_diff

    ! integer references (valid inside the OPS library) for ops_block
    type(ops_block)   :: grid2D

    !ops_dats
    type(ops_dat)     ::    d_A, d_Anew
    
    ! vars for stencils
    integer s2D_00(2) /0,0/
    type(ops_stencil) :: S2D_0pt

    integer s2D_05(10) /0,0, 1,0, -1,0, 0,1, 0,-1/
    type(ops_stencil) :: S2D_5pt

    integer d_p(2) /1,1/   !max boundary depths for the dat in the possitive direction
    integer d_m(2) /-1,-1/ !max boundary depths for the dat in the negative direction

    !size for OPS
    integer size(2)

    !base
    integer base(2) /1,1/   !this is in fortran indexing - start from 1

    real(8) :: start_time, stop_time
    
    size(1) = jmax
    size(2) = imax

    allocate ( A(1:jmax+2,1:imax+2), Anew(1:jmax+2,1:imax+2) )
    
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

    ! Initialize
    A = 0.0_8

    call cpu_time(start_time) 

    ! Set boundary conditions

    ! Bottom
    do i=1,imax+2
        A(1,i)   = 0.0_8
    end do

    ! Top
    do i=1,imax+2
        A(jmax+2,i) = 0.0_8
    end do

    ! Left
    do j=1,jmax+2
        A(j,1)   = sin(pi * j/(jmax+2))
    end do

    ! Right
    do j=1,jmax+2
        A(j,imax+2) = sin(pi * j/(jmax+2)) * exp(-pi)
    end do

    write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', jmax+2, ' x', imax+2, ' mesh'
 
    iter=0

    do i=2,imax+2
        Anew(1,i)   = 0.0_8
    end do

    do i=2,imax+2
        Anew(jmax+2,i) = 0.0_8
    end do

    do j=2,jmax+2
        Anew(j,1)   = sin(pi * j/(jmax+2))
    end do

    do j=2,jmax+2
        Anew(j,imax+2) = sin(pi * j/(jmax+2)) * exp(-pi)
    end do

    do while ( error .gt. tol .and. iter .lt. iter_max )
        error=0.0_8

        do i=2,imax+1
            do j=2,jmax+1
                Anew(j,i) = 0.25_8 * ( A(j+1,i  ) + A(j-1,i  ) + &
                                             A(j  ,i-1) + A(j  ,i+1) )
                error = max( error, abs(Anew(j,i)-A(j,i)) )
            end do
        end do
        
        do i=2,imax+1
            do j=2,jmax+1
                A(j,i) = Anew(j,i)
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

    call cpu_time(stop_time) 
    write(*,'(a,f16.7,a)')  ' completed in ', stop_time-start_time, ' seconds'

    deallocate (A,Anew)

    call ops_exit( )

end program laplace
