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

/** @brief Mesh chunk generation driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invoked the users specified chunk generator.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"
#include "ops_mpi_seq.h"

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
  printf("x_cells %d, y_cells %d x_min %d,y_min %d,x_max %d,y_max %d\n",
    x_cells, y_cells, x_min,y_min,x_max, y_max);
  ops_par_loop_mpi(generate_chunk_kernel, "generate_chunk_kernel", clover_grid, 2, rangexy,
    ops_arg_dat(vertexx,  s2D_00_P10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(vertexy,  S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(energy0,  S2D_00, "double", OPS_WRITE),
    ops_arg_dat(density0, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(xvel0,    S2D_00_P10_0P1_P1P1, "double", OPS_WRITE),
    ops_arg_dat(yvel0,    S2D_00_P10_0P1_P1P1, "double", OPS_WRITE),
    ops_arg_dat(cellx,    s2D_00_P10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celly,    S2D_00_0P1_STRID2D_Y, "double", OPS_READ));

  /*ops_print_dat_to_txtfile_core(energy0, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(density0, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(vertexx, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(vertexy, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(xx, "cloverdats.dat");*/

  ops_exit();//exit for now
  exit(0);

}
