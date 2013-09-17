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
 *  @author Wayne Gaudin
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

extern int     FIELD_DENSITY0,
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

extern FILE    *g_out, *g_in;  //Files for input and output
extern int     g_rect, g_circ, g_point; //geometry of block

extern int     number_of_states;
extern int     fields[];


/**----------OPS Vars--------------**/

//ops blocks

extern ops_block clover_grid;
extern ops_block clover_xedge;
extern ops_block clover_yedge;

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
extern ops_stencil sten_self_2D;

extern ops_stencil sten_self2D_plus1x;
extern ops_stencil sten_self2D_plus1y;
extern ops_stencil sten_self2D_minus1x;
extern ops_stencil sten_self2D_minus1y;

extern ops_stencil sten_self2D_plus1_minus1x;
extern ops_stencil sten_self2D_plus1_minus1y;

extern ops_stencil sten_self2D_minus_1_2x;
extern ops_stencil sten_self2D_minus_1_2y;

extern ops_stencil sten_self2D_plus2x;
extern ops_stencil sten_self2D_plus2y;
extern ops_stencil sten_self2D_minus2x;
extern ops_stencil sten_self2D_minus2y;

extern ops_stencil sten_self2D_plus3x;
extern ops_stencil sten_self2D_plus3y;
extern ops_stencil sten_self2D_minus3x;
extern ops_stencil sten_self2D_minus3y;

extern ops_stencil sten_self2D_plus1xy;
extern ops_stencil sten_self2D_minus1xy;

extern ops_stencil sten_self2D_4point1xy;

extern ops_stencil sten_self_stride2D_x;
extern ops_stencil sten_self_stride2D_y;

extern ops_stencil sten_self_plus1_stride2D_x;
extern ops_stencil sten_self_plus1_stride2D_y;

extern ops_stencil sten_self_minus1_stride2D_x;
extern ops_stencil sten_self_minus1_stride2D_y;

extern ops_stencil sten_self2D_plus_1_minus1_2_x;
extern ops_stencil sten_self_plus_1_minus1_2_x_stride2D_x;

extern ops_stencil sten_self_stride2D_xmax;
extern ops_stencil sten_self_nullstride2D_xmax;

#endif /* #ifndef __CLOVER_LEAF_DATA_H*/
