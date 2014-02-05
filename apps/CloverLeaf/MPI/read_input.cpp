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
  //some defailt values before read input

  test_problem = 0;
  state_max = 0;
  number_of_states = 0;

  grid = (grid_type ) xmalloc(sizeof(grid_type_core));
  grid->xmin = 0;
  grid->ymin = 0;
  grid->xmax = 100;
  grid->ymax = 100;

  grid->x_cells = 10;
  grid->y_cells = 10;

  end_time = 10.0;
  end_step = g_ibig;
  complete = FALSE;

  visit_frequency=10;
  summary_frequency=10;

  dtinit = 0.1;
  dtmax = 1.0;
  dtmin = 0.0000001;
  dtrise = 1.5;
  dtc_safe = 0.7;
  dtu_safe = 0.5;
  dtv_safe = 0.5;
  dtdiv_safe = 0.7;

  use_vector_loops = TRUE;

  //
  //need to read in the following through I/O .. hard coded below
  //

  ops_fprintf(g_out," Reading input file\n");

  #define LINESZ 1024
  char buff[LINESZ];
  FILE *fin = fopen ("clover.in", "r");
  if (fin != NULL) {
      while (fgets (buff, LINESZ, fin)) {
          char* token = strtok(buff, " =");
          while (token) {
            if(strcmp(token,"*clover\n") != 0 && strcmp(token,"*endclover\n") != 0 ) {
              //printf("token: %s ", token);
              if(strcmp(token,"initial_timestep") == 0) {
                token = strtok(NULL, " =");
                dtinit = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "initial_timestep",dtinit);
              }
              else if(strcmp(token,"max_timestep") == 0) {
                token = strtok(NULL, " =");
                dtmax = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "max_timestep",dtmax);
              }
              else if(strcmp(token,"timestep_rise") == 0) {
                token = strtok(NULL, " =");
                dtrise = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "timestep_rise",dtrise);
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
                grid->xmin = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "xmin",grid->xmin);
              }
              else if(strcmp(token,"xmax") == 0) {
                token = strtok(NULL, " =");
                grid->xmax = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "xmax",grid->xmax);
              }
              else if(strcmp(token,"ymin") == 0) {
                token = strtok(NULL, " =");
                grid->ymin = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "ymin",grid->ymin);
              }
              else if(strcmp(token,"ymax") == 0) {
                token = strtok(NULL, " =");
                grid->ymax = atof(token);
                ops_fprintf(g_out," %20s: %e\n", "ymax",grid->ymax);
              }
              else if(strcmp(token,"x_cells") == 0) {
                token = strtok(NULL, " =");
                grid->x_cells = atof(token);
                ops_fprintf(g_out," %20s: %d\n", "x_cells",grid->x_cells);
              }
              else if(strcmp(token,"y_cells") == 0) {
                token = strtok(NULL, " =");
                grid->y_cells = atof(token);
                ops_fprintf(g_out," %20s: %d\n", "y_cells",grid->y_cells);
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
              else if(strcmp(token,"test_problem") == 0) {
                token = strtok(NULL, " =");
                test_problem = atoi(token);
                ops_fprintf(g_out," %20s: %d\n", "test_problem",test_problem);
              }
              else if(strcmp(token,"state") == 0) {

                ops_fprintf(g_out,"\n");
                ops_fprintf(g_out," Reading specification for state %d\n",number_of_states+1);
                ops_fprintf(g_out,"\n");

                token = strtok(NULL, " =");
                states =  (state_type *) xrealloc(states, sizeof(state_type) * (number_of_states+1));
                states[number_of_states] = (state_type ) xmalloc(sizeof(state_type_core));
                states[number_of_states]->xvel = 0.0;
                states[number_of_states]->yvel = 0.0;


                token = strtok(NULL, " =");
                while(token) {
                  if(strcmp(token,"xvel") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->xvel = atof(token);
                    ops_fprintf(g_out,"xvel: %e\n", states[number_of_states]->xvel);
                  }
                  if(strcmp(token,"yvel") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->yvel = atof(token);
                    ops_fprintf(g_out,"yvel: %e\n", states[number_of_states]->yvel);
                  }

                  if(strcmp(token,"xmin") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->xmin = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state xmin",states[number_of_states]->xmin);
                  }
                  if(strcmp(token,"xmax") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->xmax = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state xmax",states[number_of_states]->xmax);
                  }
                  if(strcmp(token,"ymin") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->ymin = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state ymin",states[number_of_states]->ymin);
                  }
                  if(strcmp(token,"ymax") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->ymax = atof(token);
                    ops_fprintf(g_out," %20s: %e\n","state ymax",states[number_of_states]->ymax);
                  }
                  if(strcmp(token,"density") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->density = atof(token);
                    ops_fprintf(g_out," %20s: %e\n", "state density",states[number_of_states]->density);
                  }
                  if(strcmp(token,"energy") == 0) {
                    token = strtok(NULL, " =");
                    states[number_of_states]->energy = atof(token);
                    ops_fprintf(g_out," %20s: %e\n", "state energy",states[number_of_states]->energy);
                  }
                  if(strcmp(token,"geometry") == 0) {
                    token = strtok(NULL, " =");
                    if(strcmp(token,"rectangle") == 0) {
                      states[number_of_states]->geometry = g_rect;
                      ops_fprintf(g_out," %20s: %s\n","state geometry","rectangular");
                    }
                    else if(strcmp(token,"circle") == 0) {
                      states[number_of_states]->geometry = g_circ;
                      ops_fprintf(g_out," %20s: %s\n","state geometry","circular");
                    }
                    else if(strcmp(token,"point") == 0) {
                      states[number_of_states]->geometry = g_point;
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


  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Input read finished\n");
  ops_fprintf(g_out,"\n");

  field = (field_type ) xmalloc(sizeof(field_type_core));
  field->x_min = 0 ; //+2 to account for the boundary
  field->y_min = 0 ; //+2 to account for the boundary
  field->x_max = grid->x_cells ; //+2 to account for the boundary
  field->y_max = grid->y_cells ; //+2 to account for the boundary
  field->left = 0;
  field->bottom = 0;

  float dx= (grid->xmax-grid->xmin)/(float)(grid->x_cells);
  float dy= (grid->ymax-grid->ymin)/(float)(grid->y_cells);

  for(int i = 0; i < number_of_states; i++)
  {
    states[i]->xmin = states[i]->xmin + (dx/100.00);
    states[i]->ymin = states[i]->ymin + (dy/100.00);
    states[i]->xmax = states[i]->xmax - (dx/100.00);
    states[i]->ymax = states[i]->ymax - (dy/100.00);
  }
}
