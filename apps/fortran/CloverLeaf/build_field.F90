SUBROUTINE build_field()

    use OPS_Fortran_Reference
    use OPS_CONSTANTS
    USE DATA_MODULE
    USE OPS_DATA_MODULE

    use, intrinsic :: ISO_C_BINDING

    INTEGER :: x_cells, x_min, x_max 

    INTEGER :: size(2), size2(2), size3(2), size4(2), size5(2)
    INTEGER base(2) /0,0/   !array indexing - start from 0
    INTEGER d_p(2) /1,1/    !max boundary depths for the dat in the possitive direction
    INTEGER d_m(2) /-1,-1/  !max boundary depths for the dat in the negative direction

    INTEGER a2D_00(2) /0, 0/
    INTEGER a2D_00_P10(4) /0,0, 1,0/
    INTEGER a2D_00_0P1(4) /0,0, 0,1/
    
    INTEGER stride2D_x(2) /1, 0/
    INTEGER stride2D_y(2) /0, 1/

    !null array - for declaring ops dat    
    REAL(KIND=8), DIMENSION(:), ALLOCATABLE :: temp
    INTEGER, DIMENSION(:), ALLOCATABLE :: temp2

    x_cells = grid_x_cells
    x_min = field_x_min
    x_max = field_x_max
    
    !*----------------------------OPS Declarations----------------------------*
    !declare block
    call ops_decl_block(2, clover_grid, "clover grid")

    size4 = (/x_cells+6,1/)
    d_m = (/-2,0/) 
    d_p = (/2,0/)
    call ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp, vertexx, "real(8)", "vertexx")
    call ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp, vertexdx, "real(8)", "vertexdx")

    !contains x indicies from 0 to xmax+3 -- needed for initialization
    d_m = (/-2,0/)
    d_p = (/2,0/)
    call ops_decl_dat(clover_grid, 1, size4, base, d_m, d_p, temp2, xx, "integer", "xx")

    !************************************************************************************************************
    !OPS - Declare commonly used stencils
    call ops_decl_stencil( 2, 1, a2D_00, S2D_00, "00")

    call ops_decl_strided_stencil( 2, 1, a2D_00, stride2D_x, S2D_00_STRID2D_X, "s2D_00_stride2D_x")
    call ops_decl_strided_stencil( 2, 1, a2D_00, stride2D_y, S2D_00_STRID2D_Y, "s2D_00_stride2D_y")

    call ops_decl_strided_stencil( 2, 2, a2D_00_P10, stride2D_x, S2D_00_P10_STRID2D_X, "s2D_00_P10_stride2D_x")
    call ops_decl_strided_stencil( 2, 2, a2D_00_0P1, stride2D_y, S2D_00_0P1_STRID2D_Y, "s2D_00_0P1_stride2D_y")
    
    !************************************************************************************************************
    call ops_partition("2D_BLOCK_DECOMPSE")

END SUBROUTINE build_field
