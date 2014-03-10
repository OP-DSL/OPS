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
               g_ydir;

#define FIELD_DENSITY0 0
#define FIELD_DENSITY1 1
#define FIELD_ENERGY0 2
#define FIELD_ENERGY1 3
#define FIELD_PRESSURE 4
#define FIELD_VISCOSITY 5
#define FIELD_SOUNDSPEED 6
#define FIELD_XVEL0 7
#define FIELD_XVEL1 8
#define FIELD_YVEL0 9
#define FIELD_YVEL1 10
#define FIELD_VOL_FLUX_X 11
#define FIELD_VOL_FLUX_Y 12
#define FIELD_MASS_FLUX_X 13
#define FIELD_MASS_FLUX_Y 14
#define NUM_FIELDS 15

extern FILE    *g_out, *g_in;  //Files for input and output
extern int     g_rect, g_circ, g_point; //geometry of block

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
extern ops_dat vol_flux_x;
extern ops_dat vol_flux_y;
extern ops_dat mass_flux_x;
extern ops_dat mass_flux_y;
extern ops_dat xarea;
extern ops_dat yarea;

extern ops_dat work_array1;
extern ops_dat work_array2;
extern ops_dat work_array3;
extern ops_dat work_array4;
extern ops_dat work_array5;
extern ops_dat work_array6;
extern ops_dat work_array7;

extern ops_dat cellx;
extern ops_dat celly;
extern ops_dat vertexx;
extern ops_dat vertexy;
extern ops_dat celldx;
extern ops_dat celldy;
extern ops_dat vertexdx;
extern ops_dat vertexdy;

extern ops_dat xx;
extern ops_dat yy;



//commonly used stencils
extern ops_stencil S2D_00;

extern ops_stencil S2D_00_P10;
extern ops_stencil S2D_00_0P1;
extern ops_stencil S2D_00_M10;
extern ops_stencil S2D_00_0M1;

extern ops_stencil S2D_00_P10_M10;
extern ops_stencil S2D_00_0P1_0M1;

extern ops_stencil S2D_00_M10_M20;
extern ops_stencil S2D_00_0M1_0M2;

extern ops_stencil S2D_00_P20;
extern ops_stencil S2D_00_0P2;
extern ops_stencil S2D_00_M20;
extern ops_stencil S2D_00_0M2;

extern ops_stencil S2D_00_P30;
extern ops_stencil S2D_00_0P3;
extern ops_stencil S2D_00_M30;
extern ops_stencil S2D_00_0M3;

extern ops_stencil S2D_00_P40;
extern ops_stencil S2D_00_0P4;
extern ops_stencil S2D_00_M40;
extern ops_stencil S2D_00_0M4;

extern ops_stencil S2D_00_P10_0P1_P1P1;
extern ops_stencil S2D_00_M10_0M1_M1M1;

extern ops_stencil S2D_00_P10_0M1_P1M1;
extern ops_stencil S2D_00_0P1_M10_M1P1;

extern ops_stencil S2D_10_M10_01_0M1;

extern ops_stencil S2D_00_STRID2D_X;
extern ops_stencil S2D_00_STRID2D_Y;

extern ops_stencil S2D_00_P10_STRID2D_X;
extern ops_stencil S2D_00_0P1_STRID2D_Y;

extern ops_stencil S2D_00_M10_STRID2D_X;
extern ops_stencil S2D_00_0M1_STRID2D_Y;

extern ops_stencil S2D_00_P10_M10_M20;
extern ops_stencil S2D_00_P10_M10_M20_STRID2D_X;
extern ops_stencil S2D_00_0P1_0M1_0M2;
extern ops_stencil S2D_00_0P1_0M1_0M2_STRID2D_Y;

extern ops_stencil S2D_00_P10_P20_M10;
extern ops_stencil S2D_00_0P1_0P2_0M1;

#endif /* #ifndef __CLOVER_LEAF_DATA_H*/
