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
  int self2d[]  = {0,0};
  int stridey[] = {0,1};
  int stridex[] = {1,0};
  ops_stencil sten2D = ops_decl_stencil( 2, 1, self2d, "self2d");
  ops_stencil sten2D_1Dstridey = ops_decl_strided_stencil( 2, 1, self2d, stridey, "self2d");
  ops_stencil sten2D_1Dstridex = ops_decl_strided_stencil( 2, 1, self2d, stridex, "self2d");

  int self_plus1x[] = {0,0, 1,0};
  int self_plus1y[] = {0,0, 0,1};

  int strideplus1x[] = {1,0};
  int strideplus1y[] = {0,1};
  ops_stencil sten1x = ops_decl_strided_stencil( 2, 2, self_plus1x, stridex, "self_plus1x");
  ops_stencil sten1y = ops_decl_strided_stencil( 2, 2, self_plus1y, stridey, "self_plus1y");

  int four_point[]  = {0,0, 1,0, 0,1, 1,1};
  ops_stencil sten2D_4point = ops_decl_stencil( 2, 4, four_point, "sten2D_4point");
  ops_par_loop(generate_kernel, "generate_kernel", 2, rangexy,
    ops_arg_dat(vertexx, sten1x, OPS_READ),
    ops_arg_dat(vertexy, sten1y, OPS_READ),
    ops_arg_dat(energy0, sten2D, OPS_WRITE),
    ops_arg_dat(density0, sten2D, OPS_WRITE),
    ops_arg_dat(xvel0, sten2D_4point, OPS_WRITE),
    ops_arg_dat(yvel0, sten2D_4point, OPS_WRITE),
    ops_arg_dat(cellx, sten1x, OPS_READ),
    ops_arg_dat(celly, sten1y, OPS_READ));

  ops_print_dat_to_txtfile_core(density0, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(energy0, "cloverdats.dat");

}
