MODULE OPS_CONSTANTS
    
#ifdef OPS_WITH_CUDAFOR
    use cudafor
    real(kind=8), constant :: grid_xmin_opsconstant
    real(kind=8), constant :: grid_xmax_opsconstant
    integer, constant :: grid_x_cells_opsconstant

    integer, constant :: field_left_opsconstant
    integer, constant :: field_x_min_opsconstant
    integer, constant :: field_x_max_opsconstant

    real(kind=8) :: grid_xmin, grid_xmax
    integer :: grid_x_cells

    integer :: field_left
    integer :: field_x_min, field_x_max
#else
    real(kind=8) :: grid_xmin, grid_xmax
    integer :: grid_x_cells

    integer :: field_left
    integer :: field_x_min, field_x_max
#endif

END MODULE OPS_CONSTANTS
