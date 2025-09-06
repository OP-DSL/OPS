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

program random
    use OPS_Fortran_Reference
!    use OPS_Fortran_hdf5_Declarations
    use OPS_CONSTANTS

    use, intrinsic :: ISO_C_BINDING

    implicit none

    ! integer references (valid inside the OPS library) for ops_block
    type(ops_block)   :: grid2D
    
    !ops_dats
    type(ops_dat)     ::    dat0
    
    ! vars for stencils
    integer(kind=4) :: s2D_00(2) = [0,0]
    type(ops_stencil) :: S2D_0pt
    
    integer(kind=4) :: d_p(2) = [2,1]   !max boundary depths for the dat in the possitive direction
    integer(kind=4) :: d_m(2) = [-2,-1] !max boundary depths for the dat in the negative direction
    
    !size for OPS
    integer(kind=4) :: size(2)

    !base
    integer(kind=4) :: base(2) = [1,1]   !this is in fortran indexing - start from 1

    !null array - for declaring ops dat    
    real(kind=8), dimension(:), allocatable :: temp

    ! iteration range - needs to be fortran indexed here
    ! inclusive indexing for both min and max points in the range
    !.. but internally will convert to c index

    integer :: iter_range(4)
    integer(kind=4) :: iter

    !initialize and declare constants
    integer, parameter ::  x_cells = 4
    integer, parameter ::  y_cells = 4

    size(1) = x_cells
    size(2) = y_cells    

    !-----------------------OPS Initialization------------------------
    call ops_init(2)
    
    !-----------------------OPS Declarations--------------------------

    !declare block
    call ops_decl_block(2, grid2D, "grid2D")

    !declare stencils
    call ops_decl_stencil( 2, 1, s2D_00, S2D_0pt, "0pt_stencil")

    !declare ops_dat
    call ops_decl_dat(grid2D, 2, size, base, d_m, d_p, temp, dat0, "real(kind=8)", "dat0")

    call ops_partition("")

    call ops_randomgen_init(0, 0)

    DO iter = 1, 2
        call ops_fill_random_uniform(dat0)
        call ops_print_dat_to_txtfile(dat0, "data_0.txt")
    END DO

    call ops_randomgen_exit()

    iter_range = [1,4,1,4]
    call ops_par_loop(print_kernel, "print", grid2D, 2, iter_range, &
                    ops_arg_dat(dat0, 2, S2D_0pt, "real(kind=8)", OPS_READ))

    write(*,'(a)') 'This run is considered PASSED'
    call ops_exit( )

end program random
