/*Crown Copyright 2012 AWE.

 This file is part of TeaLeaf.

 TeaLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 TeaLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 TeaLeaf. If not, see http://www.gnu.org/licenses/. */

/**  @brief Holds parameters definitions
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Parameters used in the TeaLeaf are defined here.
*/


#ifndef __TEA_LEAF_DATA_H
#define __TEA_LEAF_DATA_H

#include "user_types.h"

/**----------TeaLeaf Vars/Consts--------------**/

float   g_version;
int     g_ibig;
double  g_small;
double  g_big;
int     g_name_len_max,
               g_xdir,
               g_ydir;

FILE    *g_out, *g_in;  //Files for input and output
int     g_rect, g_circ, g_point; //geometry of block

int     number_of_states;
int     fields[NUM_FIELDS];


/**----------OPS Vars--------------**/

//ops blocks

ops_block tea_grid;

//ops dats
ops_dat density       ;
ops_dat energy0       ;
ops_dat energy1       ;
ops_dat u             ;
ops_dat u0            ;
ops_dat vector_r      ;
ops_dat vector_rstore ;
ops_dat vector_rtemp  ;
ops_dat vector_Mi     ;
ops_dat vector_w      ;
ops_dat vector_z      ;
ops_dat vector_utemp  ;
ops_dat vector_Kx     ;
ops_dat vector_Ky     ;
ops_dat vector_p      ;
ops_dat vector_sd     ;
ops_dat tri_cp        ;
ops_dat tri_bfp       ;
ops_dat cellx         ;
ops_dat celly         ;
ops_dat vertexx       ;
ops_dat vertexy       ;
ops_dat celldx        ;
ops_dat celldy        ;
ops_dat vertexdx      ;
ops_dat vertexdy      ;
ops_dat volume        ;
ops_dat xarea         ;
ops_dat yarea         ;

ops_dat xx;
ops_dat yy;



//commonly used stencils
ops_stencil S2D_00;

ops_stencil S2D_00_P10;
ops_stencil S2D_00_0P1;
ops_stencil S2D_00_M10;
ops_stencil S2D_00_0M1;

ops_stencil S2D_00_P10_M10;
ops_stencil S2D_00_0P1_0M1;

ops_stencil S2D_00_M10_M20;
ops_stencil S2D_00_0M1_0M2;

ops_stencil S2D_00_M10_0M1;

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

ops_stencil S2D_00_P10_STRID2D_X;
ops_stencil S2D_00_0P1_STRID2D_Y;

ops_stencil S2D_00_M10_STRID2D_X;
ops_stencil S2D_00_0M1_STRID2D_Y;

ops_stencil S2D_00_P10_M10_M20;
ops_stencil S2D_00_P10_M10_STRID2D_X;
ops_stencil S2D_00_0P1_0M1_0M2;
ops_stencil S2D_00_0P1_0M1_STRID2D_Y;

ops_stencil S2D_00_P10_P20_M10;
ops_stencil S2D_00_0P1_0P2_0M1;

ops_stencil S2D_00_0M1_M10_P10_0P1;

ops_reduction red_local_dt;
ops_reduction red_xl_pos;
ops_reduction red_yl_pos;
ops_reduction red_vol;
ops_reduction red_mass;
ops_reduction red_ie;
ops_reduction red_ke;
ops_reduction red_press;
ops_reduction red_output;
ops_reduction red_temp;

#endif /* #ifndef __TEA_LEAF_DATA_H*/
