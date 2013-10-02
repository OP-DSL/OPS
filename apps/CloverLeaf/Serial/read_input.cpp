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

/** @brief Reads the user input
 * @author Wayne Gaudin
 * @details Reads and parses the user input from the processed file and sets
 * the variables used in the generation phase. Default values are also set
 * here.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

void read_input()
{
  int state, stat, state_max, n;
  int number_of_states;
  double dx, dy;
  int grid_xmin = 0.0, grid_ymin = 0.0, grid_xmax = 100.0, grid_ymax = 100.0;
  int grid_x_cells = 10, grid_y_cells = 10;
  double end_time = 10.0;
  int end_step = g_ibig;
  int complete = 0;

  double dtinit = 0.1;
  double dtmax = 1.0;
  double dtmin = 0.0000001;
  double dtrise = 1.5;
  double dtc_safe = 0.7;
  double dtu_safe = 0.5;
  double dtv_safe = 0.5;
  double dtdiv_safe = 0.7;

  ops_fprintf(g_out," Reading input file\n");
  ops_fprintf(g_out,"\n");

  //
  //For now hard code the default problem
  //
  dtinit=0.04;
  dtmax=dtinit;
  dtrise=1.5;
  grid_x_cells=10;
  grid_y_cells=2;
  grid_xmin=0.0;
  grid_ymin=0.0;
  grid_xmax=10.0;
  grid_ymax=2.0;
  end_time=3.0;

  state_max = 2;
  number_of_states = state_max;

  if(number_of_states < 1)
  {
    ops_printf(" read_input , No states defined\n");
    exit(-1);
  }

  states = (state_type *)xmalloc(number_of_states * sizeof(state_type));
  for(int i = 0; i < number_of_states; i++)
  {
    state_type s = ( state_type ) xmalloc ( sizeof ( state_type_core ) );
    s->defined = FALSE;
    s->energy = 0.0;
    s->density = 0.0;
    s->xvel = 0.0;
    s->yvel = 0.0;
    states[i] = s;
  }

  states[0]->density=0.2;
  states[0]->energy=1.0;

  states[1]->density=1.0;
  states[1]->energy=2.5;
  states[1]->geometry=g_rect;
  states[1]->xmin=0.0;
  states[1]->xmax=5.0;
  states[1]->ymin=0.0;
  states[1]->ymax=2.0;

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Input read finished.\n");
  ops_fprintf(g_out,"\n");


  dx=(grid_xmax-grid_xmin)/grid_x_cells;
  dy=(grid_ymax-grid_ymin)/grid_y_cells;
  for(int i = 1; i < number_of_states; i++)
  {
    states[i]->xmin=states[i]->xmin+(dx/100.0);
    states[i]->ymin=states[i]->ymin+(dy/100.0);
    states[i]->xmax=states[i]->xmax-(dx/100.0);
    states[i]->ymax=states[i]->ymax-(dy/100.0);
  }


}
