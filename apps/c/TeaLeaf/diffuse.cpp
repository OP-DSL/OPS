/*Crown Copyright 2014 AWE.

 This file is part of TeaLeaf.

 TeaLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 TeaLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 TeaLeaf. If not, see http://www.gnu.org/licenses/. */

// @brief Controls the main diffusion cycle.
// @author Istvan Reguly, David Beckingsale, Wayne Gaudin
// @details Controls the top level cycle, invoking all the drivers and checks
// for outputs and completion.


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include <ops_seq.h>


#include "data.h"
#include "definitions.h"

#include "tea_leaf.h"

        //These two need to be kept consistent with update_halo
int     CHUNK_LEFT    = 0,
        CHUNK_RIGHT   = 1,
        CHUNK_BOTTOM  = 2,
        CHUNK_TOP     = 3,
        EXTERNAL_FACE = -1;

state_type * states; //global variable holding state info

grid_type grid; //global variable holding global grid info

field_type field; //global variable holding info of fields

int step ;
int advect_x; //logical
int error_condition;
int test_problem;
int profiler_on;
int state_max;
int complete; //logical

double dtold, dt, clover_time, dtinit, dtmin, dtmax, eps, tl_ch_cg_epslim;

int x_min, y_min, x_max, y_max, x_cells, y_cells, max_iters, coefficient;

int tl_ch_cg_presteps,
   tl_check_result
  ,tl_preconditioner_type
  ,reflective_boundary
  ,tl_ppcg_inner_steps
  ,tl_use_chebyshev
  ,tl_use_cg
  ,tl_use_ppcg
  ,tl_ppcg_active
  ,tl_use_jacobi;

int verbose_on = 0;

double currtime, end_time;
int end_step;
int visit_frequency;
int summary_frequency;
int tiling_frequency;

int jdt, kdt;

void process_profile() {}

void diffuse()
{
  double ct0, et0, ct1,et1, timerstart, wall_clock, step_clock;
  double grind_time, cells, rstep, step_time, step_grind;

  ops_timers(&ct0, &timerstart);


  while (true) {
    ops_timers(&ct0, &step_time);

    step = step + 1;

    timestep();

    tea_leaf();

    currtime = currtime + dt;

    if(summary_frequency != 0)
      if((step%summary_frequency) == 0)
        field_summary();

    ops_timers(&ct0, &et0);
    wall_clock = et0 - timerstart;
    step_clock = et0 - step_time;
    ops_printf(" Wall clock %.15lf\n", wall_clock);
    ops_fprintf(g_out," Wall clock %.15lf\n", wall_clock);
    cells = grid.x_cells * grid.y_cells;
    rstep = step;
    grind_time = wall_clock/(rstep * cells);
    step_grind = step_clock/cells;
    ops_printf(" Average time per cell   %-10.15E\n", grind_time);
    ops_fprintf(g_out," Average time per cell   %-10.15E\n", grind_time);
    ops_printf(" Step time per cell      %-10.15E\n", step_grind);
    ops_fprintf(g_out," Step time per cell      %-10.15E\n", step_grind);

    if((clover_time+g_small) > end_time || (step >= end_step)) {
      complete=1;
      field_summary();
      ops_fprintf(g_out,"\n\n Calculation complete\n");
      ops_fprintf(g_out,"\n Tea is finishing\n");
      ops_printf(" Wall clock %.15lf\n", wall_clock);
      ops_fprintf(g_out," Wall clock %.15lf\n", wall_clock);
      break;
    }
    
  }

  ops_timers(&ct1, &et1);

  if(profiler_on == 1) {
    ops_timing_output(std::cout); // print output to STDOUT
    //ops_timing_output(g_out);
    process_profile();
  }

  ops_printf("\nTotal Wall time %-10.15E\n",et1-timerstart);
  ops_fprintf(g_out,"\nTotal Wall time %-10.15E\n",et1-timerstart);

  fclose(g_out);
}
