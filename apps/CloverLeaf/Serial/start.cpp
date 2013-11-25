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

/** @brief Main set up routine
 * @author Wayne Gaudin
 * @details Invokes the mesh decomposer and sets up chunk connectivity. It then
 * allocates the communication buffers and call the chunk initialisation and
 * generation routines. It calls the equation of state to calculate initial
 * pressure before priming the halo cells and writing an initial field summary.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"


#include "data.h"
#include "definitions.h"



void initialise_chunk();
void generate();
void build_field();
void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void field_summary();


void start()
{
  time  = 0.0;
  step  = 0;
  dtold = dtinit;
  dt    = dtinit;

  build_field();

  /**---------------------------Initialize Chunks----------------------------**/

  initialise_chunk();

  /**---------------------------Generating Chunks----------------------------**/

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Generating chunks\n");
  ops_fprintf(g_out,"\n");

  generate();

  advect_x = TRUE;

  /**------------------------------ideal_gas---------------------------------**/

  ideal_gas(FALSE);

  /**-----------------------------update_halo--------------------------------**/

  //Prime all halo data for the first step
  fields[FIELD_DENSITY0]  = 1;
  fields[FIELD_ENERGY0]   = 1;
  fields[FIELD_PRESSURE]  = 1;
  fields[FIELD_VISCOSITY] = 1;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_XVEL0]     = 1;
  fields[FIELD_YVEL0]     = 1;
  fields[FIELD_XVEL1]     = 1;
  fields[FIELD_YVEL1]     = 1;

  update_halo(fields, 2);

  ops_fprintf(g_out,"\n");
  ops_fprintf(g_out," Problem initialised and generated\n");
  ops_fprintf(g_out,"\n");

  /**----------------------------field_summary-------------------------------**/

  field_summary();

}
