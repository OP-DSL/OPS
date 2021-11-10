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
#include "ops_seq.h"


#include "data.h"
#include "definitions.h"



void read_input();

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
    ops_fprintf(g_in," state 2 density=1.0 energy=2.5 geometry=cuboid xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0 zmin=0.0 zmax=2.0\n");
    ops_fprintf(g_in," x_cells=10\n");
    ops_fprintf(g_in," y_cells=2\n");
    ops_fprintf(g_in," z_cells=2\n");
    ops_fprintf(g_in," xmin=0.0\n");
    ops_fprintf(g_in," ymin=0.0\n");
    ops_fprintf(g_in," zmin=0.0\n");
    ops_fprintf(g_in," xmax=10.0\n");
    ops_fprintf(g_in," ymax=2.0\n");
    ops_fprintf(g_in," zmax=2.0\n");
    ops_fprintf(g_in," initial_timestep=0.04\n");
    ops_fprintf(g_in," timestep_rise=1.05\n");
    ops_fprintf(g_in," max_timestep=0.04\n");
    ops_fprintf(g_in," end_step=75\n");
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

  read_input();

  ops_fprintf(g_out," Starting the calculation\n");
  fclose(g_in);
}
