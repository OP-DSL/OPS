/*Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**  @brief Holds parameters definitions
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Parameters used in the CloverLeaf are defined here.
*/


#ifndef __CLOVER_LEAF_DATA_H
#define __CLOVER_LEAF_DATA_H

/**----------Cloverleaf Vars/Consts--------------**/

extern float   g_version;
extern int     g_ibig;
extern double  g_small;
extern double  g_big;
extern int     g_name_len_max,
               g_xdir,
               g_ydir,
               g_zdir;

/*extern int     FIELD_DENSITY0,
        FIELD_DENSITY1,
        FIELD_ENERGY0,
        FIELD_ENERGY1,
        FIELD_PRESSURE,
        FIELD_VISCOSITY,
        FIELD_SOUNDSPEED,
        FIELD_XVEL0,
        FIELD_XVEL1,
        FIELD_YVEL0,
        FIELD_YVEL1,
        FIELD_VOL_FLUX_X,
        FIELD_VOL_FLUX_Y,
        FIELD_MASS_FLUX_X,
        FIELD_MASS_FLUX_Y,
        NUM_FIELDS;
*/
extern FILE    *g_out, *g_in;  //Files for input and output
extern int     g_cube, g_sphe, g_point; //geometry of block

extern int     number_of_states;
extern int     fields[];


/**----------OPS Vars--------------**/

//ops blocks

extern ops_block clover_grid;

//ops dats
extern ops_dat density0;
extern ops_dat density1;
extern ops_dat energy0;
extern ops_dat energy1;
extern ops_dat pressure;
extern ops_dat viscosity;
extern ops_dat soundspeed;
extern ops_dat volume;

extern ops_dat xvel0;
extern ops_dat xvel1;
extern ops_dat yvel0;
extern ops_dat yvel1;
extern ops_dat zvel0;
extern ops_dat zvel1;
extern ops_dat vol_flux_x;
extern ops_dat vol_flux_y;
extern ops_dat vol_flux_z;
extern ops_dat mass_flux_x;
extern ops_dat mass_flux_y;
extern ops_dat mass_flux_z;
extern ops_dat xarea;
extern ops_dat yarea;
extern ops_dat zarea;

extern ops_dat work_array1;
extern ops_dat work_array2;
extern ops_dat work_array3;
extern ops_dat work_array4;
extern ops_dat work_array5;
extern ops_dat work_array6;
extern ops_dat work_array7;

extern ops_dat cellx;
extern ops_dat celly;
extern ops_dat cellz;
extern ops_dat vertexx;
extern ops_dat vertexy;
extern ops_dat vertexz;
extern ops_dat celldx;
extern ops_dat celldy;
extern ops_dat celldz;
extern ops_dat vertexdx;
extern ops_dat vertexdy;
extern ops_dat vertexdz;

extern ops_dat xx;
extern ops_dat yy;
extern ops_dat zz;



//commonly used stencils
extern ops_stencil S3D_000;
 
extern ops_stencil S3D_000_P100;
extern ops_stencil S3D_000_0P10;
extern ops_stencil S3D_000_00P1;

extern ops_stencil S3D_000_M100;
extern ops_stencil S3D_000_0M10;
extern ops_stencil S3D_000_00M1;

extern ops_stencil S3D_000_f0M1M1;
extern ops_stencil S3D_000_fM10M1;
extern ops_stencil S3D_000_fM1M10;

extern ops_stencil S3D_000_f0P1P1;
extern ops_stencil S3D_000_fP10P1;
extern ops_stencil S3D_000_fP1P10;

extern ops_stencil S3D_000_fP1P1P1;

extern ops_stencil S3D_000_fP1M1M1;
extern ops_stencil S3D_000_fM1P1M1;
extern ops_stencil S3D_000_fM1M1P1;

extern ops_stencil S3D_000_fM1P1P1;
extern ops_stencil S3D_000_fP1M1P1;
extern ops_stencil S3D_000_fP1P1M1;

extern ops_stencil S3D_000_fM1M1M1;

extern ops_stencil S3D_000_P100_P200_M100;
extern ops_stencil S3D_000_0P10_0P20_0M10;
extern ops_stencil S3D_000_00P1_00P2_00M1;
extern ops_stencil S3D_000_P100_M100_M200;
extern ops_stencil S3D_000_0P10_0M10_0M20;
extern ops_stencil S3D_000_00P1_00M1_00M2;

extern ops_stencil S3D_P100_M100_0P10_0M10_00P1_00M1;

extern ops_stencil S3D_000_P200;
extern ops_stencil S3D_000_0P20;
extern ops_stencil S3D_000_00P2;
extern ops_stencil S3D_000_M200;
extern ops_stencil S3D_000_0M20;
extern ops_stencil S3D_000_00M2;
extern ops_stencil S3D_000_P300;
extern ops_stencil S3D_000_0P30;
extern ops_stencil S3D_000_00P3;
extern ops_stencil S3D_000_M300;
extern ops_stencil S3D_000_0M30;
extern ops_stencil S3D_000_00M3;
extern ops_stencil S3D_000_P400;
extern ops_stencil S3D_000_0P40;
extern ops_stencil S3D_000_00P4;
extern ops_stencil S3D_000_M400;
extern ops_stencil S3D_000_0M40;
extern ops_stencil S3D_000_00M4;

extern ops_stencil S3D_000_STRID3D_X;
extern ops_stencil S3D_000_STRID3D_Y;
extern ops_stencil S3D_000_STRID3D_Z;

extern ops_stencil S3D_000_P100_STRID3D_X;
extern ops_stencil S3D_000_0P10_STRID3D_Y;
extern ops_stencil S3D_000_00P1_STRID3D_Z;

extern ops_stencil S3D_000_P100_M100_STRID3D_X;
extern ops_stencil S3D_000_0P10_0M10_STRID3D_Y;
extern ops_stencil S3D_000_00P1_00M1_STRID3D_Z;

extern ops_stencil S3D_000_P100_M100_M200_STRID3D_X;
extern ops_stencil S3D_000_0P10_0M10_0M20_STRID3D_Y;
extern ops_stencil S3D_000_00P1_00M1_00M2_STRID3D_Z;

extern ops_reduction red_local_dt;
extern ops_reduction red_xl_pos;
extern ops_reduction red_yl_pos;
extern ops_reduction red_zl_pos;
extern ops_reduction red_vol;
extern ops_reduction red_mass;
extern ops_reduction red_ie;
extern ops_reduction red_ke;
extern ops_reduction red_press;
extern ops_reduction red_output;

#endif /* #ifndef __CLOVER_LEAF_DATA_H*/
