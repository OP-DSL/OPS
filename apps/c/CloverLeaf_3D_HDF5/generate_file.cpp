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

/** @brief CloverLeaf HDF5 file generation code to be used with CLoverLeaf_3D_HDF5
  * @author Gihan Mudalige
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


// OPS header file
#include "ops_seq.h"

// Cloverleaf constants
#include "data.h"

// Cloverleaf definitions
#include "definitions.h"


// Cloverleaf functions
void initialise();
void initialise_chunk();
void generate_hdf5();
void build_field_hdf5();

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
        g_ydir = 2,
        g_zdir = 3;

int     number_of_states;

        //These two need to be kept consistent with update_halo
int     CHUNK_LEFT    = 1,
        CHUNK_RIGHT   = 2,
        CHUNK_BOTTOM  = 3,
        CHUNK_TOP     = 4,
        CHUNK_BACK    = 5,
        CHUNK_FRONT   = 6,
        EXTERNAL_FACE = -1;

FILE    *g_out, *g_in;  //Files for input and output

int     g_cube=1,
        g_sphe=2,
        g_point=3;

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

int fields[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

double dtold, dt, clover_time, dtinit, dtmin, dtmax, dtrise, dtu_safe, dtv_safe, dtw_safe, dtc_safe,
       dtdiv_safe, dtc, dtu, dtv, dtdiv;

//int x_min, y_min, z_min, x_max, y_max, z_max, x_cells, y_cells, z_cells;

double end_time;
int end_step;
int visit_frequency;
int summary_frequency;
int use_vector_loops;

int jdt, kdt, ldt;


#include "cloverleaf_ops_vars.h"


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  /**-------------------------- OPS Initialisation --------------------------**/

  // OPS initialisation
  ops_init(argc,argv,1);
  ops_printf(" Clover version %f\n", g_version);


  /**---------------------initialize and generate chunk----------------------**/

  initialise();

  //initialize sizes using global values
  ops_decl_const("g_small", 1, "double", &g_small );
  ops_decl_const("g_big", 1, "double", &g_big );
  ops_decl_const("dtc_safe", 1, "double", &dtc_safe );
  ops_decl_const("dtu_safe", 1, "double", &dtu_safe );
  ops_decl_const("dtv_safe", 1, "double", &dtv_safe );
  ops_decl_const("dtw_safe", 1, "double", &dtw_safe );
  ops_decl_const("dtdiv_safe", 1, "double", &dtdiv_safe );
  ops_decl_const("field", 1, "field_type", &field );
  ops_decl_const("grid", 1, "grid_type", &grid );
  ops_decl_const("states",number_of_states, "state_type", states );
  ops_decl_const("number_of_states",1,"int",&number_of_states);
  ops_decl_const("g_sphe",1,"int",&g_sphe);
  ops_decl_const("g_point",1,"int",&g_point);
  ops_decl_const("g_cube",1,"int",&g_cube);
  ops_decl_const("dt", 1, "double", &dt );

  /**--------------------------decompose 2D grid ----------------------------**/
  if (ops_is_root()) {
    ops_fprintf(g_out," Setting up initial geometry\n");
    ops_fprintf(g_out,"\n");
  }

  clover_time  = 0.0;
  step  = 0;
  dtold = dtinit;
  dt    = dtinit;

  build_field_hdf5();

  /**---------------------------Initialize Chunks----------------------------**/

  initialise_chunk();

  /**---------------------------Generating Chunks----------------------------**/

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Generating chunks\n");
  ops_fprintf(g_out,"\n");

  generate_hdf5();

  fclose(g_out);
  ops_exit();
}
