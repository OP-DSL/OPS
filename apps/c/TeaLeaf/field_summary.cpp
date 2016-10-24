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

/** @brief Print field summary
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details The user specified field summary kernel is invoked here. A summation
 *  across all mesh chunks is then performed and the information outputed.
 *  If the run is a test problem, the final result is compared with the expected
 *  result and the difference output.
 *  Note the reference solution is the value returned from an Intel compiler with
 *  ieee options set on a single core crun.
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
#include "field_summary_kernel.h"

void field_summary()
{
  double qa_diff;

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  double vol= 0.0 , mass = 0.0, ie = 0.0, temp = 0.0;

  ops_par_loop(field_summary_kernel, "field_summary_kernel", tea_grid, 2, rangexy_inner,
      ops_arg_dat(volume, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(density, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(energy1, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(u, 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_vol, 1, "double", OPS_INC),
      ops_arg_reduce(red_mass, 1, "double", OPS_INC),
      ops_arg_reduce(red_ie, 1, "double", OPS_INC),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  //printf("mass = %lf\n",mass);
  ops_reduction_result(red_vol,&vol);
  ops_reduction_result(red_mass,&mass);
  ops_reduction_result(red_ie,&ie);
  ops_reduction_result(red_temp,&temp);


  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out,"\n Time %lf\n",clover_time);
  ops_fprintf(g_out,"              %-10s  %-10s  %-15s  %-10s  %-s\n",
  " Volume"," Mass"," Density"," Internal Energy","Temperature");
  ops_fprintf(g_out," step:   %3d   %-10.3E  %-10.3E  %-15.3E  %-10.3E  %-.3E",
          step, vol, mass, mass/vol, ie, temp);

  if(complete == 1) {
    if(test_problem>0) {
      if (test_problem == 1)
        qa_diff = fabs((100.0 * (temp / 157.550841832793)) - 100.0);
      if (test_problem == 2) // tea_bm_short.in
        qa_diff = fabs((100.0 * (temp / 116.067951160930)) - 100.0);
      if (test_problem == 3)
        qa_diff = fabs((100.0 * (temp / 95.4865103390698)) - 100.0);
      if (test_problem == 4)
        qa_diff = fabs((100.0 * (temp / 166.838315378708)) - 100.0);
      if (test_problem == 5)
        qa_diff = fabs((100.0 * (temp / 116.482111627676)) - 100.0);
      if (test_problem == 6)
        qa_diff = fabs((100.0 * (temp / 103.88639125996923)) - 100.0);
      ops_printf("Test problem %3d is within   %-10.7E%% of the expected solution\n",test_problem, qa_diff);
      ops_fprintf(g_out,"\nTest problem %3d is within   %10.7E%% of the expected solution\n",test_problem, qa_diff);
      if(qa_diff < 0.001) {
        ops_printf(" This test is considered PASSED\n");
        ops_fprintf(g_out," This test is considered PASSED\n");
      }
      else
      {
        ops_printf(" This test is considered NOT PASSED\n");
        ops_fprintf(g_out," This test is considered NOT PASSED\n");
      }
    }
  }
  fflush(g_out);

  //ops_exit();//exit for now
  //exit(0);

}
