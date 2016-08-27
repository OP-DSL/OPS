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
#include <ops_seq.h>

#include "data.h"
#include "definitions.h"

void read_input()
{
  //some defailt values before read input

  test_problem = 0;
  
  state_max = 0;

//  number_of_states = 0;

  grid.xmin = 0;
  grid.ymin = 0;
  grid.xmax = 100;
  grid.ymax = 100;

  grid.x_cells = 10;
  grid.y_cells = 10;

  end_time = 10.0;
  end_step = g_ibig;
  complete = FALSE;

  visit_frequency=10;
  summary_frequency=10;

  dtinit = 0.1;
  //dtmax = 1.0;
  max_iters = 1000;
  eps  = 1e-10;

  coefficient = CONDUCTIVITY;

  profiler_on  = 0;

  tl_ch_cg_presteps = 25;
  tl_ch_cg_epslim = 1.0;
  tl_check_result = 0;
  tl_preconditioner_type = TL_PREC_NONE;
  reflective_boundary = 0;
  tl_ppcg_inner_steps = -1;
  tl_use_chebyshev = 0;
  tl_use_cg = 0;
  tl_use_ppcg = 0;
  tl_use_jacobi = 1;
  verbose_on = 0;

  //
  //need to read in the following through I/O .. hard coded below
  //

  ops_fprintf(g_out," Reading input file\n");

  #define LINESZ 1024
  char buff[LINESZ];
  FILE *fin = fopen ("tea.in", "r");
  if (fin != NULL) {
      while (fgets (buff, LINESZ, fin)) {
          char* token = strtok(buff, " =");
          while (token) {
            if(strcmp(token,"*tea\n") != 0 && strcmp(token,"*endtea\n") != 0 ) {
              //printf("token: %s ", token);
              if(strcmp(token,"initial_timestep") == 0) {
                token = strtok(NULL, " =");
                dtinit = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "initial_timestep",dtinit);
              }
              else if(strcmp(token,"end_time") == 0) {
                token = strtok(NULL, " =");
                end_time = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "end_time",end_time);
              }
              else if(strcmp(token,"end_step") == 0) {
                token = strtok(NULL, " =");
                end_step = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "end_step",end_step);
              }
              else if(strcmp(token,"xmin") == 0) {
                token = strtok(NULL, " =");
                grid.xmin = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "xmin",grid.xmin);
              }
              else if(strcmp(token,"xmax") == 0) {
                token = strtok(NULL, " =");
                grid.xmax = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "xmax",grid.xmax);
              }
              else if(strcmp(token,"ymin") == 0) {
                token = strtok(NULL, " =");
                grid.ymin = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "ymin",grid.ymin);
              }
              else if(strcmp(token,"ymax") == 0) {
                token = strtok(NULL, " =");
                grid.ymax = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "ymax",grid.ymax);
              }
              else if(strcmp(token,"x_cells") == 0) {
                token = strtok(NULL, " =");
                grid.x_cells = atof(token);
                ops_fprintf(g_out," %20s: %d\n", "x_cells",grid.x_cells);
              }
              else if(strcmp(token,"y_cells") == 0) {
                token = strtok(NULL, " =");
                grid.y_cells = atof(token);
                ops_fprintf(g_out," %20s: %d\n", "y_cells",grid.y_cells);
              }
              else if(strcmp(token,"visit_frequency") == 0) {
                token = strtok(NULL, " =");
                visit_frequency = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "visit_frequency",visit_frequency);
              }
              else if(strcmp(token,"summary_frequency") == 0) {
                token = strtok(NULL, " =");
                summary_frequency = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "summary_frequency",summary_frequency);
              }
              else if(strcmp(token,"tl_ch_cg_presteps") == 0) {
                token = strtok(NULL, " =");
                tl_ch_cg_presteps = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "tl_ch_cg_presteps",tl_ch_cg_presteps);
              }
              else if(strcmp(token,"tl_ppcg_inner_steps") == 0) {
                token = strtok(NULL, " =");
                tl_ppcg_inner_steps = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "tl_ppcg_inner_steps",tl_ppcg_inner_steps);
              }
              else if(strcmp(token,"tl_ch_cg_epslim") == 0) {
                token = strtok(NULL, " =");
                tl_ch_cg_epslim = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "tl_ch_cg_epslim",tl_ch_cg_epslim);
              }
              else if(strcmp(token,"tl_check_result") == 0) {
                token = strtok(NULL, " =");
                tl_check_result = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "tl_check_result",tl_check_result);
              }
              else if(strcmp(token,"tl_preconditioner_type") == 0) {
                token = strtok(NULL, " =");
                if (strcmp(token,"TL_PREC_NONE") == 0)
                  tl_preconditioner_type = TL_PREC_NONE;
                else if (strcmp(token,"TL_PREC_JAC_DIAG") == 0)
                  tl_preconditioner_type = TL_PREC_JAC_DIAG;
                else if (strcmp(token,"TL_PREC_JAC_BLOCK") == 0)
                  tl_preconditioner_type = TL_PREC_JAC_BLOCK;
                else {
                  ops_printf("Unrecognized preconditioner type in input file\n");
                  exit(-1);
                }
                ops_fprintf(g_out," %20s: %d\n", "tl_preconditioner_type",tl_preconditioner_type);
              }
              else if(strcmp(token,"verbose_on") == 0) {
                token = strtok(NULL, " =");
                verbose_on = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "verbose_on",verbose_on);
              }
              else if(strcmp(token,"tl_use_jacobi") == 0) {
                tl_use_jacobi = 1;
                tl_use_chebyshev = 0;
                tl_use_ppcg = 0;
                tl_use_cg = 0;
              }
              else if(strcmp(token,"tl_use_chebyshev") == 0) {
                tl_use_jacobi = 0;
                tl_use_chebyshev = 1;
                tl_use_ppcg = 0;
                tl_use_cg = 0;
              }
              else if(strcmp(token,"tl_use_ppcg") == 0) {
                tl_use_jacobi = 0;
                tl_use_chebyshev = 0;
                tl_use_ppcg = 1;
                tl_use_cg = 0;
              }
              else if(strcmp(token,"tl_use_cg") == 0) {
                tl_use_jacobi = 0;
                tl_use_chebyshev = 0;
                tl_use_ppcg = 0;
                tl_use_cg = 1;
              }
              else if(strcmp(token,"reflective_boundary") == 0) {
                token = strtok(NULL, " =");
                reflective_boundary = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "reflective_boundary",reflective_boundary);
              }
              else if(strcmp(token,"test_problem") == 0) {
                token = strtok(NULL, " =");
                test_problem = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "test_problem",test_problem);
              }
              else if(strcmp(token,"profiler_on") == 0) {
                token = strtok(NULL, " =");
                profiler_on = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "profiler_on",profiler_on);
              }
              else if(strcmp(token,"tl_max_iters") == 0) {
                token = strtok(NULL, " =");
                tl_max_iters = atoi(token);
              }
              else if(strcmp(token,"tl_eps") == 0) {
                token = strtok(NULL, " =");
                tl_eps = atof(token);
              }
              else if(strcmp(token,"tl_coefficient_density") == 0) {
                coefficient = CONDUCTIVITY;
                ops_fprintf(g_out," %s\n", "Diffusion coefficient density");
              }
              else if(strcmp(token,"tl_coefficient_inverse_density") == 0) {
                coefficient = RECIP_CONDUCTIVITY;
                ops_fprintf(g_out," %s\n", "iffusion coefficient reciprocal density");
              }
              else if(strcmp(token,"state") == 0) {

                ops_fprintf(g_out,"\n");
                ops_fprintf(g_out," Reading specification for state %d\n",number_of_states+1);
                ops_fprintf(g_out,"\n");

                token = strtok(NULL, " =");
                states =  (state_type *) xrealloc(states, sizeof(state_type) * (number_of_states+1));
                states[number_of_states].defined = 1;

                token = strtok(NULL, " =");
                while(token) {
                  if(strcmp(token,"xmin") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].xmin = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state xmin",states[number_of_states].xmin);
                  }
                  if(strcmp(token,"xmax") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].xmax = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state xmax",states[number_of_states].xmax);
                  }
                  if(strcmp(token,"ymin") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].ymin = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state ymin",states[number_of_states].ymin);
                  }
                  if(strcmp(token,"ymax") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].ymax = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state ymax",states[number_of_states].ymax);
                  }
                  if(strcmp(token,"density") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].density = atof(token);
                    ops_fprintf(g_out," %20s: %e\n", "state density",states[number_of_states].density);
                  }
                  if(strcmp(token,"energy") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states].energy = atof(token);
                    ops_fprintf(g_out," %20s: %e\n", "state energy",states[number_of_states].energy);
                  }
                  if(strcmp(token,"geometry") == 0) {
                    token = strtok(NULL, " =");
                    if(strcmp(token,"rectangle") == 0) {
                      states[number_of_states].geometry = g_rect;
                      ops_fprintf(g_out," %20s: %s\n","state geometry","rectangular");
                    }
                    else if(strcmp(token,"circle") == 0) {
                      states[number_of_states].geometry = g_circ;
                      ops_fprintf(g_out," %20s: %s\n","state geometry","circular");
                    }
                    else if(strcmp(token,"point") == 0) {
                      states[number_of_states].geometry = g_point;
                      ops_fprintf(g_out," %20s: %s\n","state geometry","point");
                    }
                  }

                  token = strtok(NULL, " =");

                }

                number_of_states++;
                ops_fprintf(g_out,"\n");
              }
            }
            token = strtok(NULL, " =");

          }
      }
      fclose (fin);
  }

  if(number_of_states == 0) {
    ops_printf("read_input, No states defined.\n");
    exit(-1);
  }

  // Simple guess - better than a default of 10
  if (tl_ppcg_inner_steps == -1) {
    tl_ppcg_inner_steps = 4*sqrt(sqrt((double)(grid.x_cells*grid.y_cells)));
    ops_fprintf(g_out," %20s: %d\n", "tl_ppcg_inner_steps",tl_ppcg_inner_steps);
  }

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Input read finished\n");
  ops_fprintf(g_out,"\n");

  //field = (field_type ) xmalloc(sizeof(field_type_core));
  field.x_min = 0 +2; //+2 to account for the boundary
  field.y_min = 0 +2; //+2 to account for the boundary
  field.x_max = grid.x_cells +2; //+2 to account for the boundary
  field.y_max = grid.y_cells +2; //+2 to account for the boundary
  field.left = 0;
  field.bottom = 0;

  float dx= (grid.xmax-grid.xmin)/(float)(grid.x_cells);
  float dy= (grid.ymax-grid.ymin)/(float)(grid.y_cells);

  for(int i = 0; i < number_of_states; i++)
  {
    states[i].xmin = states[i].xmin + (dx/100.00);
    states[i].ymin = states[i].ymin + (dy/100.00);
    states[i].xmax = states[i].xmax - (dx/100.00);
    states[i].ymax = states[i].ymax - (dy/100.00);
  }
}
