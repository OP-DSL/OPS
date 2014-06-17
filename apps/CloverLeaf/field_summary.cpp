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
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "field_summary_kernel.h"

extern"C" {
void fortfunc_(double* time,int *step,double *vol,double *mass,double *press,double *ie,double *ke, char* buf);
}

void ideal_gas(int predict);

void field_summary()
{
  double qa_diff;

  //initialize sizes using global values
  int x_cells = grid.x_cells;
  int y_cells = grid.y_cells;
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  //call ideal_gas again here
  ideal_gas(FALSE);

  double vol= 0.0 , mass = 0.0, ie = 0.0, ke = 0.0, press = 0.0;

  ops_par_loop(field_summary_kernel, "field_summary_kernel", clover_grid, 2, rangexy_inner,
      ops_arg_dat(volume, S2D_00, "double", OPS_READ),
      ops_arg_dat(density0, S2D_00, "double", OPS_READ),
      ops_arg_dat(energy0, S2D_00, "double", OPS_READ),
      ops_arg_dat(pressure, S2D_00, "double", OPS_READ),
      ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_gbl(&vol, 1, "double", OPS_INC),
      ops_arg_gbl(&mass, 1, "double", OPS_INC),
      ops_arg_gbl(&ie, 1, "double", OPS_INC),
      ops_arg_gbl(&ke, 1, "double", OPS_INC),
      ops_arg_gbl(&press, 1, "double", OPS_INC));

  char buffer[350];
  if(ops_is_root()){
    fortfunc_(&clover_time,&step,&vol,&mass,&press,&ie,&ke,buffer);
  }

  //ops_fprintf(g_out,"\n");
  //ops_fprintf(g_out,"\n Time   %1.14E\n",clover_time);
  //ops_fprintf(g_out,"                 %11s     %11s     %11s     %11s %15s %15s %15s\n",
  //" Volume"," Mass"," Density"," Pressure","Internal Energy","Kinetic Energy","Total Energy");
  //ops_fprintf(g_out," step:   %4d     %10.3E      %10.3E      %10.3E      %10.3E %15.3E %15.3E %15.3E",
  //        step, vol, mass, mass/vol, press/vol, ie, ke, ie+ke);
  ops_fprintf(g_out,"\n%s\n",buffer);

  if(complete == TRUE) {
    if(test_problem == 1) {
      qa_diff=fabs((100.0*(ke/1.82280367310258))-100.0);
      ops_printf("\nTest problem 1 is within %3.15E %% of the expected solution\n",qa_diff);
      ops_fprintf(g_out,"\nTest problem 1 is within %3.15E %% of the expected solution\n",qa_diff);
      if(qa_diff < 0.001) {
        ops_printf("This test is considered PASSED\n");
        ops_fprintf(g_out,"This test is considered PASSED\n");
      }
      else
      {
        ops_printf("This test is considered NOT PASSED\n");
        ops_fprintf(g_out,"This test is considered NOT PASSED\n");
      }
    }
  }
  fflush(g_out);

  //ops_exit();//exit for now
  //exit(0);

}
