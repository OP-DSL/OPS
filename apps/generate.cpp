/* Crown Copyright 2012 AWE.

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

/** @brief Top level initialisation routine
 *  @author Wayne Gaudin
 *  @details Checks for the user input and either invokes the input reader or
 *  switches to the internal test problem. It processes the input and strips
 *  comments before writing a final input file.
 *  It then calls the start routine.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "generate_chunk_kernel.h"

void generate()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop(generate_kernel, "generate_kernel", 2, rangexy,
    ops_arg_dat(vertexx,  sten_self_plus1_stride2D_x, OPS_READ),
    ops_arg_dat(vertexy,  sten_self_plus1_stride2D_y, OPS_READ),
    ops_arg_dat(energy0,  sten_self_2D, OPS_WRITE),
    ops_arg_dat(density0, sten_self_2D, OPS_WRITE),
    ops_arg_dat(xvel0,    sten_self2D_plus1xy, OPS_WRITE),
    ops_arg_dat(yvel0,    sten_self2D_plus1xy, OPS_WRITE),
    ops_arg_dat(cellx,    sten_self_plus1_stride2D_x, OPS_READ),
    ops_arg_dat(celly,    sten_self_plus1_stride2D_y, OPS_READ));



}
