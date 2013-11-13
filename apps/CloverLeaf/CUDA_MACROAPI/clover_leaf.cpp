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

/** @brief CloverLeaf top level program: Invokes the main cycle
  * @author Wayne Gaudin, converted to OPS by Gihan Mudalige
  * @details CloverLeaf in a proxy-app that solves the compressible Euler
  *  Equations using an explicit finite volume method on a Cartesian grid.
  *  The grid is staggered with internal energy, density and pressure at cell
  *  centres and velocities on cell vertices.

  *  A second order predictor-corrector method is used to advance the solution
  *  in time during the Lagrangian phase. A second order advective remap is then
  *  carried out to return the mesh to an orthogonal state.
  *
  *  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
  *  work on a mesh with varying spacing to keep it relevant to it's parent code.
  *  For this reason, optimisations should only be carried out on the software
  *  that do not change the underlying numerical method. For example, the
  *  volume, though constant for all cells, should remain array and not be
  *  converted to a scalar.
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing Structured mesh applications
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// OPS header file
#include "ops_seq_opt.h"

// Cloverleaf constants
#include "data.h"

// Cloverleaf definitions
#include "definitions.h"


// Cloverleaf functions
void initialise();
void field_summary();
void timestep();
void PdV(int predict);
void accelerate();
void flux_calc();
void advection(int);
void reset_field();



/******************************************************************************
* Initialize Global constants and variables
/******************************************************************************/


/**----------Cloverleaf Vars/Consts--------------**/

float   g_version = 1.0;
int     g_ibig = 640000;
double  g_small = 1.0e-16;
double  g_big  = 1.0e+21;
int     g_name_len_max = 255 ,
        g_xdir = 1,
        g_ydir = 2;

int     number_of_states;

        //These two need to be kept consistent with update_halo
int     CHUNK_LEFT    = 1,
        CHUNK_RIGHT   = 2,
        CHUNK_BOTTOM  = 3,
        CHUNK_TOP     = 4,
        EXTERNAL_FACE = -1;

int     FIELD_DENSITY0   = 0,
        FIELD_DENSITY1   = 1,
        FIELD_ENERGY0    = 2,
        FIELD_ENERGY1    = 3,
        FIELD_PRESSURE   = 4,
        FIELD_VISCOSITY  = 5,
        FIELD_SOUNDSPEED = 6,
        FIELD_XVEL0      = 7,
        FIELD_XVEL1      = 8,
        FIELD_YVEL0      = 9,
        FIELD_YVEL1      =10,
        FIELD_VOL_FLUX_X =11,
        FIELD_VOL_FLUX_Y =12,
        FIELD_MASS_FLUX_X=13,
        FIELD_MASS_FLUX_Y=14,
        NUM_FIELDS       =15;

FILE    *g_out, *g_in;  //Files for input and output

int     g_rect=1,
        g_circ=2,
        g_point=3;

state_type * states; //global variable holding state info

grid_type grid; //global variable holding global grid info

field_type field; //global variable holding info of fields

int step ;
int advect_x; //logical
int error_condition;
int test_problem;
int state_max;
int complete; //logical

int fields[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

double dtold, dt, clover_time, dtinit, dtmin, dtmax, dtrise, dtu_safe, dtv_safe, dtc_safe,
       dtdiv_safe, dtc, dtu, dtv, dtdiv;

double end_time;
int end_step;
int visit_frequency;
int summary_frequency;
int use_vector_loops;

int jdt, kdt;

#include "cloverleaf_ops_vars.h"


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- OPS Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,5);
  ops_printf(" Clover version %f\n", g_version);


  /**---------------------initialize and generate chunk----------------------**/

  initialise();


  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  //fields = (int *)malloc(NUM_FIELDS*sizeof(int));
  //for(int i=0; i<NUM_FIELDS;i++) fields[i] = 0;

  /***************************************************************************
  **-----------------------------hydro loop---------------------------------**
  /**************************************************************************/
  double ct0, ct1, et0, et1;
  ops_timers_core(&ct0, &et0);

  while(1) {

    step = step + 1;


    timestep();

    if(step == 1) {
      //ops_print_dat_to_txtfile_core(work_array1, "cloverdats.dat");
      //ops_print_dat_to_txtfile_core(xvel1, "cloverdats.dat");
      //exit(0);
    }

    PdV(TRUE);

    accelerate();

    //ops_print_dat_to_txtfile_core(volume, "cloverdats.dat");
    //exit(0);


    PdV(FALSE);

    flux_calc();

    advection(step);

    reset_field();

    if (advect_x == TRUE) advect_x = FALSE;
    else advect_x = TRUE;

    clover_time = clover_time + dt;

    if(summary_frequency != 0)
      if((step%summary_frequency) == 0)
        field_summary();

    if((clover_time+g_small) > end_time || (step >= end_step)) {
      complete=TRUE;
      field_summary();
      break;
    }

      if(step == 20) {
        //ops_print_dat_to_txtfile_core(viscosity, "cloverdats.dat");
        //ops_print_dat_to_txtfile_core(xvel1, "cloverdats.dat");
        //exit(0);
        break;
      }
  }

  ops_timers_core(&ct1, &et1);
  ops_printf("\nTotal Wall time %lf\n",et1-et0);

  fclose(g_out);
  ops_exit();
}
