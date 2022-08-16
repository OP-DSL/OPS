MODULE OPS_DATA_MODULE
    use OPS_Fortran_Reference

    USE, INTRINSIC :: ISO_C_BINDING
         
    IMPLICIT NONE 

    !clover leaf OPS vars
    TYPE(ops_block) :: clover_grid

    TYPE(ops_dat) :: vertexx, vertexdx
    TYPE(ops_dat) :: xx

    TYPE(ops_stencil) :: S2D_00
    TYPE(ops_stencil) :: S2D_00_STRID2D_X, S2D_00_STRID2D_Y
    TYPE(ops_stencil) :: S2D_00_P10_STRID2D_X, S2D_00_0P1_STRID2D_Y
    
end module ops_data_module     
