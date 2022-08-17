SUBROUTINE initialise_chunk()

    use OPS_Fortran_Reference    
    use OPS_CONSTANTS
    USE DATA_MODULE
    USE OPS_DATA_MODULE
    
    use, intrinsic :: ISO_C_BINDING

    INTEGER :: x_cells, x_min, x_max
    
    INTEGER :: rangex(4), rangey(4), rangexy(4), rangefull(4)

    x_cells = grid_x_cells
    y_cells = grid_y_cells
    x_min = field_x_min
    x_max = field_x_max
    y_min = field_y_min
    y_max = field_y_max

    rangex = [x_min-2, x_max+3, 0, 0]

    call ops_par_loop(initialise_chunk_kernel_xx, "initialise_chunk_kernel_xx", clover_grid, 2, rangex, &
                    & ops_arg_dat(xx, 1, S2D_00_STRID2D_X, "integer", OPS_WRITE), &
                    & ops_arg_idx())

    call ops_par_loop(initialise_chunk_kernel_x, "initialise_chunk_kernel_x", clover_grid, 2, rangex, &
                    & ops_arg_dat(vertexx, 1, S2D_00_STRID2D_X, "real(8)", OPS_WRITE), &
                    & ops_arg_dat(xx, 1, S2D_00_STRID2D_X, "integer", OPS_READ), &
                    & ops_arg_dat(vertexdx, 1, S2D_00_STRID2D_X, "real(8)", OPS_WRITE))
    
    rangey = [0, 0, y_min-2, y_max+3]


    rangexy = [x_min-2, x_max+2, y_min-2, y_max+2]

    

END SUBROUTINE initialise_chunk

