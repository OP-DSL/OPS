PROGRAM clover_leaf

    use OPS_Fortran_Reference
    use OPS_CONSTANTS
    use DATA_MODULE
    use OPS_DATA_MODULE

    use, intrinsic :: ISO_C_BINDING

    ! profiling
    REAL(kind=c_double) :: startTime = 0
    REAL(kind=c_double) :: endTime = 0



    !-----------------------OPS Initialization------------------------
    call ops_init(2)

    ! start timer
    call ops_timers ( startTime )
        
    CALL initialise( )

#ifdef OPS_WITH_CUDAFOR
    gridx_min_opsconstant = grid_xmin
    grid_xmax_opsconstant = grid_xmax
    grid_x_cells_opsconstant = grid_x_cells
    field_left_opsconstant = field_left
    field_x_min_opsconstant = field_x_min
    field_x_max_opsconstant = field_x_max
#endif

    call ops_decl_const("grid_xmin", 1, "double", grid_xmin)
    call ops_decl_const("grid_xmax", 1, "double", grid_xmax)
    call ops_decl_const("grid_x_cells", 1, "int", grid_x_cells)
    call ops_decl_const("field_left", 1, "int", field_left)
    call ops_decl_const("field_x_min", 1, "int", field_x_min)
    call ops_decl_const("field_x_max", 1, "int", field_x_max)

    CALL start( )
    
    !hydro cycle
    !DO
    !    step = step + 1
    !    time = time + dt
    !END DO  !end of hydro cycle loop    



    CALL ops_timers( endTime )
    WRITE(*,'(a,f16.7,a)')  ' completed in ', endTime - startTime, ' seconds'
    
    call ops_exit( )

END PROGRAM clover_leaf
