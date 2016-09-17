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

extern double dtold, dt, clover_time, dtinit, dtmin, dtmax, eps, tl_ch_cg_epslim;

extern int x_min, y_min, x_max, y_max, x_cells, y_cells, max_iters, coefficient;

extern int tl_ch_cg_presteps,
   tl_check_result
  ,tl_preconditioner_type
  ,reflective_boundary
  ,tl_ppcg_inner_steps
  ,tl_use_chebyshev
  ,tl_use_cg
  ,tl_use_ppcg
  ,tl_use_jacobi
  ,tl_ppcg_active
  ,verbose_on;

extern int tl_max_iters;

extern int jdt, kdt;

extern double time, end_time;
        //These two need to be kept consistent with update_halo
extern int     CHUNK_LEFT ,
        CHUNK_RIGHT   ,
        CHUNK_BOTTOM  ,
        CHUNK_TOP     ,
        EXTERNAL_FACE ;
#endif /* __CLOVER_LEAF_DEFINITIONS_H */
