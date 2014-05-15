
  //
  //ops blocks
  //

  ops_block clover_grid;

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
  ops_dat zvel0;
  ops_dat zvel1;
  ops_dat vol_flux_x;
  ops_dat vol_flux_y;
  ops_dat vol_flux_z;
  ops_dat mass_flux_x;
  ops_dat mass_flux_y;
  ops_dat mass_flux_z;
  ops_dat xarea;
  ops_dat yarea;
  ops_dat zarea;

  ops_dat work_array1;
  ops_dat work_array2;
  ops_dat work_array3;
  ops_dat work_array4;
  ops_dat work_array5;
  ops_dat work_array6;
  ops_dat work_array7;

  ops_dat cellx;
  ops_dat celly;
  ops_dat cellz;
  ops_dat vertexx;
  ops_dat vertexy;
  ops_dat vertexz;
  ops_dat celldx;
  ops_dat celldy;
  ops_dat celldz;
  ops_dat vertexdx;
  ops_dat vertexdy;
  ops_dat vertexdz;

  ops_dat xx;
  ops_dat yy;
  ops_dat zz;

  //
  //Declare commonly used stencils
  //
  ops_stencil S3D_000;

  ops_stencil S3D_000_P100;
  ops_stencil S3D_000_0P10;
  ops_stencil S3D_000_00P1;

  ops_stencil S3D_000_M100;
  ops_stencil S3D_000_0M10;
  ops_stencil S3D_000_00M1;

  ops_stencil S3D_000_f0M1M1;
  ops_stencil S3D_000_fM10M1;
  ops_stencil S3D_000_fM1M10;

  ops_stencil S3D_000_f0P1P1;
  ops_stencil S3D_000_fP10P1;
  ops_stencil S3D_000_fP1P10;

  ops_stencil S3D_000_fP1P1P1;

  ops_stencil S3D_000_fP1M1M1;
  ops_stencil S3D_000_fM1P1M1;
  ops_stencil S3D_000_fM1M1P1;
  
  ops_stencil S3D_000_fM1P1P1;
  ops_stencil S3D_000_fP1M1P1;
  ops_stencil S3D_000_fP1P1M1;

  ops_stencil S3D_000_fM1M1M1;

  ops_stencil S3D_000_P100_P200_M100;
  ops_stencil S3D_000_0P10_0P20_0M10;
  ops_stencil S3D_000_00P1_00P2_00M1;
  ops_stencil S3D_000_P100_M100_M200;
  ops_stencil S3D_000_0P10_0M10_0M20;
  ops_stencil S3D_000_00P1_00M1_00M2;
  
  ops_stencil S3D_P100_M100_0P10_0M10_00P1_00M1;

  ops_stencil S3D_000_P200;
  ops_stencil S3D_000_0P20;
  ops_stencil S3D_000_00P2;
  ops_stencil S3D_000_M200;
  ops_stencil S3D_000_0M20;
  ops_stencil S3D_000_00M2;
  ops_stencil S3D_000_P300;
  ops_stencil S3D_000_0P30;
  ops_stencil S3D_000_00P3;
  ops_stencil S3D_000_M300;
  ops_stencil S3D_000_0M30;
  ops_stencil S3D_000_00M3;
  ops_stencil S3D_000_P400;
  ops_stencil S3D_000_0P40;
  ops_stencil S3D_000_00P4;
  ops_stencil S3D_000_M400;
  ops_stencil S3D_000_0M40;
  ops_stencil S3D_000_00M4;

  ops_stencil stride3D_x;
  ops_stencil stride3D_y;
  ops_stencil stride3D_z;

  ops_stencil S3D_000_STRID3D_X;
  ops_stencil S3D_000_STRID3D_Y;
  ops_stencil S3D_000_STRID3D_Z;

  ops_stencil S3D_000_P100_STRID3D_X;
  ops_stencil S3D_000_0P10_STRID3D_Y;
  ops_stencil S3D_000_00P1_STRID3D_Z;

  ops_stencil S3D_000_P100_M100_STRID3D_X;
  ops_stencil S3D_000_0P10_0M10_STRID3D_Y;
  ops_stencil S3D_000_00P1_00M1_STRID3D_Z;
  
  ops_stencil S3D_000_P100_M100_M200_STRID3D_X;
  ops_stencil S3D_000_0P10_0M10_0M20_STRID3D_Y;
  ops_stencil S3D_000_00P1_00M1_00M2_STRID3D_Z;