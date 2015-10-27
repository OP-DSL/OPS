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

/** @brief Holds the high level data types
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details The high level data types used to store the mesh and field data
 *  are defined here.

 *  Also the global variables used for defining the input and controlling the
 *  scheme are defined here.
*/

#ifndef __CLOVER_LEAF_DEFINITIONS_H
#define __CLOVER_LEAF_DEFINITIONS_H

#define FALSE 0
#define TRUE 1
#include "user_types.h"

extern state_type * states;

extern grid_type grid;

extern field_type field;

extern int step;

extern int advect_x;
extern int error_condition;
extern int test_problem;
extern int profiler_on;
extern int state_max;
extern int complete;

extern double end_time;
extern int end_step;
extern int visit_frequency;
extern int summary_frequency;
extern int use_vector_loops;

extern double dtold, dt, clover_time, dtinit, dtmin, dtmax, dtrise, dtu_safe, dtv_safe, dtw_safe, dtc_safe,
       dtdiv_safe, dtc, dtu, dtv, dtdiv;

//extern int x_min, y_min, z_min, x_max, y_max, z_max, x_cells, y_cells, z_cells;

extern int jdt, kdt, ldt;

#endif
