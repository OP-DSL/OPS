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
#include "initialise_chunk_kernel.h"

void initialise()
{

  char out_file[] = "clover.out";
  char in_file[] = "clover.in";

  if ((g_out = fopen(out_file,"w")) == NULL) {
    ops_printf("can't open file %s\n",out_file); exit(-1);
  }

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out,"Clover version %f\n", g_version);
  ops_printf("Output file %s opened. All output will go there\n", out_file);
  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Clover will run from the following input:-\n");
  ops_fprintf(g_out,"\n");

  if ((g_in = fopen(in_file,"r")) == NULL) {
    g_in = fopen(in_file,"w+");
    ops_fprintf(g_in,"*clover\n");
    ops_fprintf(g_in," state 1 density=0.2 energy=1.0\n");
    ops_fprintf(g_in," state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0\n");
    ops_fprintf(g_in," x_cells=10\n");
    ops_fprintf(g_in," y_cells=2\n");
    ops_fprintf(g_in," xmin=0.0\n");
    ops_fprintf(g_in," ymin=0.0\n");
    ops_fprintf(g_in," xmax=10.0\n");
    ops_fprintf(g_in," ymax=2.0\n");
    ops_fprintf(g_in," initial_timestep=0.04\n");
    ops_fprintf(g_in," timestep_rise=1.5\n");
    ops_fprintf(g_in," max_timestep=0.04\n");
    ops_fprintf(g_in," end_time=3.0\n");
    ops_fprintf(g_in," test_problem 1\n");
    ops_fprintf(g_in,"*endclover\n");
    fclose(g_in);
    g_in = fopen(in_file,"r");
  }

  char line[80];
  while(fgets(line, 80, g_in) != NULL)
  {
    ops_fprintf(g_out,"%s", line);
  }

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Initialising and generating\n");
  ops_fprintf(g_out,"\n");


  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int self[] = {0,0};
  ops_stencil sten1 = ops_decl_stencil( 2, 1, self, "self");

  int rangex[] = {x_min-2, x_max+3, 0, 1};
  ops_par_loop(initialise_chunk_kernel_x, "initialise_chunk_kernel_x", 2, rangex,
               ops_arg_dat(vertexx, sten_self_2D, OPS_WRITE),
               ops_arg_dat(xx, sten_self_2D, OPS_READ),
               ops_arg_dat(vertexdx, sten_self_2D, OPS_WRITE));

  int rangey[] = {0, 1, y_min-2, y_max+3};
  ops_par_loop(initialise_chunk_kernel_y, "initialise_chunk_kernel_y", 2, rangey,
               ops_arg_dat(vertexy, sten_self_2D, OPS_WRITE),
               ops_arg_dat(yy, sten_self_2D, OPS_READ),
               ops_arg_dat(vertexdy, sten_self_2D, OPS_WRITE));

  rangex[0] = x_min-2; rangex[1] = x_max+2; rangex[2] = 0; rangex[3] = 1;
  ops_par_loop(initialise_chunk_kernel_cellx, "initialise_chunk_kernel_cellx", 2, rangex,
               ops_arg_dat(vertexx, sten_self2D_plus1x, OPS_READ),
               ops_arg_dat(cellx, sten_self_2D, OPS_WRITE),
               ops_arg_dat(celldx, sten_self_2D, OPS_WRITE));

  rangey[0] = 0; rangey[1] = 1; rangey[2] = y_min-2; rangey[3] = y_max+2;
  ops_par_loop(initialise_chunk_kernel_celly, "initialise_chunk_kernel_celly", 2, rangey,
               ops_arg_dat(vertexy, sten_self2D_plus1y, OPS_READ),
               ops_arg_dat(celly, sten_self_2D, OPS_WRITE),
               ops_arg_dat(celldy, sten_self_2D, OPS_WRITE));

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop(initialise_volume_xarea_yarea, "initialise_volume_xarea_yarea", 2, rangexy,
    ops_arg_dat(volume, sten_self_2D, OPS_WRITE),
    ops_arg_dat(celldy, sten_self_stride2D_y, OPS_READ),
    ops_arg_dat(xarea, sten_self_2D, OPS_WRITE),
    ops_arg_dat(celldx, sten_self_stride2D_x, OPS_READ),
    ops_arg_dat(yarea, sten_self_2D, OPS_WRITE));

}
