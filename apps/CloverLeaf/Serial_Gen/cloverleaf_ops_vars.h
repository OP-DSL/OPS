
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

  ops_stencil S2D_00_P30;
  ops_stencil S2D_00_0P3;
  ops_stencil S2D_00_M30;
  ops_stencil S2D_00_0M3;

  ops_stencil S2D_00_P40;
  ops_stencil S2D_00_0P4;
  ops_stencil S2D_00_M40;
  ops_stencil S2D_00_0M4;

  ops_stencil S2D_00_P10_0P1_P1P1;
  ops_stencil S2D_00_M10_0M1_M1M1;

  ops_stencil S2D_00_P10_0M1_P1M1;
  ops_stencil S2D_00_0P1_M10_M1P1;

  ops_stencil S2D_10_M10_01_0M1;

  ops_stencil S2D_00_STRID2D_X;
  ops_stencil S2D_00_STRID2D_Y;

  ops_stencil s2D_00_P10_STRID2D_X;
  ops_stencil S2D_00_0P1_STRID2D_Y;

  ops_stencil S2D_00_M10_STRID2D_X;
  ops_stencil S2D_00_0M1_STRID2D_Y;

  ops_stencil S2D_00_P10_M10_M20;
  ops_stencil S2D_00_P10_M10_M20_STRID2D_X;
  ops_stencil S2D_00_0P1_0M1_0M2;
  ops_stencil S2D_00_0P1_0M1_0M2_STRID2D_Y;

  ops_stencil S2D_00_P10_P20_M10;
  ops_stencil S2D_00_0P1_0P2_0M1;


  ops_stencil sten_self_stride2D_xmax;
  ops_stencil sten_self_nullstride2D_xmax;
  ops_stencil sten_self_stride2D_ymax;
  ops_stencil sten_self_nullstride2D_ymax;
