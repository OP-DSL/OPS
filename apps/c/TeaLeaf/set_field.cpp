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

/** @brief Set field kernel
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Copies all of the final start of step filed data to the end of step data.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
 #define OPS_2D
#include <ops_seq.h>

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "set_field_kernels.h"

void set_field()
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  ops_par_loop(set_field_kernel, "set_field_kernel", tea_grid, 2, rangexy_inner,
      ops_arg_dat(energy0, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(energy1, 1, S2D_00, "double", OPS_WRITE));

}
