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
 *  @author Wayne Gaudin
 *  @details The high level data types used to store the mesh and field data
 *  are defined here.

 *  Also the global variables used for defining the input and controlling the
 *  scheme are defined here.
*/

#ifndef __CLOVER_LEAF_DEFINITIONS_H
#define __CLOVER_LEAF_DEFINITIONS_H

#define FALSE 0
#define TRUE 1

typedef struct
{
      int defined;  //logical
      double density,
             energy,
             xvel,
             yvel;
      int geometry;
      double xmin,
             xmax,
             ymin,
             ymax,
             radius;
} state_type_core;
typedef state_type_core * state_type;

extern state_type * states;

typedef struct
{
  double  xmin, ymin, xmax, ymax;
  int x_cells, y_cells;
} grid_type_core;
typedef grid_type_core * grid_type;

extern grid_type grid;

typedef struct
{
  int left, right, bottom, top ,left_boundary, right_boundary,
      bottom_boundary, top_boundary;
  int x_min, y_min, x_max ,y_max;
} field_type_core;
typedef field_type_core * field_type;

extern field_type field;

extern int step;

extern int advect_x;
extern int error_condition;
extern int test_problem;
extern int complete;

extern double end_time;
extern int end_step;

extern double dtold, dt, time, dtinit, dtmin, dtmax, dtrise, dtu_safe, dtv_safe, dtc_safe,
       dtdiv_safe, dtc, dtu, dtv, dtdiv;

extern int jdt, kdt;


#endif /* __CLOVER_LEAF_DEFINITIONS_H */
