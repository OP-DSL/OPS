SUBROUTINE initialise_chunk_kernel_x(vertexx, xx, vertexdx)
    IMPLICIT NONE

    REAL(KIND=8), DIMENSION(1) :: vertexx
    INTEGER, DIMENSION(1), INTENT(IN) :: xx
    REAL(KIND=8), DIMENSION(1) :: vertexdx

    INTEGER :: x_min
    REAL(KIND=8) :: min_x, d_x

    x_min = field_x_min - 2
    d_x = (grid_xmax - grid_xmin)/float(grid_x_cells)
    min_x = grid_xmin + d_x*field_left

    vertexx(OPS_ACC1(0,0)) = min_x + d_x * (xx(OPS_ACC2(0,0)) - x_min)
    vertexdx(OPS_ACC3(0,0)) = float(d_x)

END SUBROUTINE initialise_chunk_kernel_x

SUBROUTINE initialise_chunk_kernel_xx(xx, idx)
    IMPLICIT NONE

    INTEGER, DIMENSION(1) :: xx
    INTEGER, DIMENSION(2), INTENT(IN) :: idx

    xx(OPS_ACC1(0,0)) = idx(1)-2

END SUBROUTINE initialise_chunk_kernel_xx

