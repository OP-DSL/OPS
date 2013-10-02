
  //
  //ops blocks
  //

  ops_block clover_grid;
  ops_block clover_xedge;
  ops_block clover_yedge;

  //
  //ops dats
  //
  ops_dat density0;
  ops_dat density1;
  ops_dat energy0;
  ops_dat energy1;
  ops_dat pressure;
  ops_dat viscosity;
  ops_dat soundspeed;
  ops_dat volume;

  ops_dat xvel0;
  ops_dat xvel1;
  ops_dat yvel0;
  ops_dat yvel1;
  ops_dat vol_flux_x;
  ops_dat vol_flux_y;
  ops_dat mass_flux_x;
  ops_dat mass_flux_y;
  ops_dat xarea;
  ops_dat yarea;

  ops_dat work_array1;
  ops_dat work_array2;
  ops_dat work_array3;
  ops_dat work_array4;
  ops_dat work_array5;
  ops_dat work_array6;
  ops_dat work_array7;

  ops_dat cellx;
  ops_dat celly;
  ops_dat vertexx;
  ops_dat vertexy;
  ops_dat celldx;
  ops_dat celldy;
  ops_dat vertexdx;
  ops_dat vertexdy;

  ops_dat xx;
  ops_dat yy;

  //
  //Declare commonly used stencils
  //
  ops_stencil S2D_00;

  ops_stencil S2D_00_P10;
  ops_stencil S2D_00_0P1;
  ops_stencil S2D_00_M10;
  ops_stencil S2D_00_0M1;

  ops_stencil S2D_00_P10_M10;
  ops_stencil S2D_00_0P1_0M1;

  ops_stencil S2D_00_M10_M20;
  ops_stencil S2D_00_0M1_0M2;

  ops_stencil S2D_00_P20;
  ops_stencil S2D_00_0P2;
  ops_stencil S2D_00_M20;
  ops_stencil S2D_00_0M2;

  ops_stencil sten_self2D_plus3x;
  ops_stencil sten_self2D_plus3y;
  ops_stencil sten_self2D_minus3x;
  ops_stencil sten_self2D_minus3y;

  ops_stencil sten_self2D_plus4x;
  ops_stencil sten_self2D_plus4y;
  ops_stencil sten_self2D_minus4x;
  ops_stencil sten_self2D_minus4y;

  ops_stencil sten_self2D_plus1xy;
  ops_stencil sten_self2D_minus1xy;

  ops_stencil sten_self2D_plus1x_minus1y;
  ops_stencil sten_self2D_plus1y_minus1x;

  ops_stencil sten_self2D_4point1xy;

  ops_stencil sten_self_stride2D_x;
  ops_stencil sten_self_stride2D_y;

  ops_stencil sten_self_plus1_stride2D_x;
  ops_stencil sten_self_plus1_stride2D_y;

  ops_stencil sten_self_minus1_stride2D_x;
  ops_stencil sten_self_minus1_stride2D_y;

  ops_stencil sten_self2D_plus_1_minus1_2_x;
  ops_stencil sten_self_plus_1_minus1_2_x_stride2D_x;
  ops_stencil sten_self2D_plus_1_minus1_2_y;
  ops_stencil sten_self_plus_1_minus1_2_y_stride2D_y;

  ops_stencil sten_self2D_plus_1_2_minus1x;
  ops_stencil sten_self2D_plus_1_2_minus1y;


  ops_stencil sten_self_stride2D_xmax;
  ops_stencil sten_self_nullstride2D_xmax;
  ops_stencil sten_self_stride2D_ymax;
  ops_stencil sten_self_nullstride2D_ymax;
