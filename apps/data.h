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

extern int      number_of_states;

#endif /* #ifndef __CLOVER_LEAF_DATA_H*/
