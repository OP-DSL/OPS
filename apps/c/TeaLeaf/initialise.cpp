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
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
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
#include <ops_seq.h>


#include "data.h"
#include "definitions.h"

void read_input();
void start();

void initialise()
{
  char out_file[] = "tea.out";
  char in_file[] = "tea.in";

  if ((g_out = fopen(out_file,"w")) == NULL) {
    ops_printf("can't open file %s\n",out_file); exit(-1);
  }

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out,"Tea version %f\n", g_version);
  ops_printf("Output file %s opened. All output will go there\n", out_file);
  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Tea will run from the following input:-\n");
  ops_fprintf(g_out,"\n");

  if ((g_in = fopen(in_file,"r")) == NULL) {
    g_in = fopen(in_file,"w+");
    ops_fprintf(g_in,"*tea\n");
    ops_fprintf(g_in," state 1 density=100.0 energy=0.0001\n");
    ops_fprintf(g_in," state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0\n");
    ops_fprintf(g_in," state 3 density=0.1 energy=0.1 geometry=rectangle xmin=1.0 xmax=6.0 ymin=1.0 ymax=2.0\n");
    ops_fprintf(g_in," state 4 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=6.0 ymin=1.0 ymax=8.0\n");
    ops_fprintf(g_in," state 5 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=10.0 ymin=7.0 ymax=8.0\n");
    ops_fprintf(g_in," x_cells=10\n");
    ops_fprintf(g_in," y_cells=10\n");
    ops_fprintf(g_in," xmin=0.0\n");
    ops_fprintf(g_in," ymin=0.0\n");
    ops_fprintf(g_in," xmax=10.0\n");
    ops_fprintf(g_in," ymax=10.0\n");
    ops_fprintf(g_in," initial_timestep=0.004\n");
    ops_fprintf(g_in," end_step=10\n");
    ops_fprintf(g_in," tl_max_iters=1000\n");
    ops_fprintf(g_in," tl_use_jacobi\n");
    ops_fprintf(g_in," tl_eps=1e-15\n");
    ops_fprintf(g_in," test_problem 1\n");
    ops_fprintf(g_in,"*endtea\n");
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

  read_input();

  step = 0;

  start();

  ops_fprintf(g_out," Starting the calculation\n");
  fclose(g_in);
}
