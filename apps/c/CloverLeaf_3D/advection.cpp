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

/** @brief Top level advection driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Controls the advection step and invokes required communications.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

void update_halo(int* fields, int depth);
void advec_cell(int sweep_number, int direction);
void advec_mom(int which_vel, int sweep_number, int dir);

void advection(int step)
{
  int sweep_number, direction;
  int xvel,yvel,zvel;

  sweep_number = 1;
  if(advect_x == TRUE) direction = g_xdir;
  if(!(advect_x == TRUE)) direction = g_zdir;

  xvel = g_xdir;
  yvel = g_ydir;
  zvel = g_zdir;

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_XVEL1]     = 0;
  fields[FIELD_YVEL1]     = 0;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_ZVEL1]     = 0;
  fields[FIELD_VOL_FLUX_X] = 1;
  fields[FIELD_VOL_FLUX_Y] = 1;
  fields[FIELD_VOL_FLUX_Z] = 1;
  fields[FIELD_MASS_FLUX_X] = 0;
  fields[FIELD_MASS_FLUX_Y] = 0;
  fields[FIELD_MASS_FLUX_Z] = 0;
  update_halo(fields,2);

  advec_cell(sweep_number, direction);

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_XVEL1]     = 1;
  fields[FIELD_YVEL1]     = 1;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_ZVEL1]     = 1;
  fields[FIELD_VOL_FLUX_X] = 0;
  fields[FIELD_VOL_FLUX_Y] = 0;
  fields[FIELD_VOL_FLUX_Z] = 0;
  fields[FIELD_MASS_FLUX_X] = (direction==g_xdir);
  fields[FIELD_MASS_FLUX_Y] = (direction==g_ydir);
  fields[FIELD_MASS_FLUX_Z] = (direction==g_zdir);
  update_halo(fields,2);


  advec_mom(xvel, sweep_number, direction);
  advec_mom(yvel, sweep_number, direction);
  advec_mom(zvel, sweep_number, direction);

  sweep_number = 2;
  direction = g_ydir;

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_XVEL1]     = 0;
  fields[FIELD_YVEL1]     = 0;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_ZVEL1]     = 0;
  fields[FIELD_VOL_FLUX_X] = 1;
  fields[FIELD_VOL_FLUX_Y] = 1;
  fields[FIELD_VOL_FLUX_Z] = 1;
  fields[FIELD_MASS_FLUX_X] = 0;
  fields[FIELD_MASS_FLUX_Y] = 0;
  fields[FIELD_MASS_FLUX_Z] = 0;
  update_halo(fields,2);

  advec_cell(sweep_number,direction);

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_XVEL1]     = 1;
  fields[FIELD_YVEL1]     = 1;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_ZVEL1]     = 1;
  fields[FIELD_VOL_FLUX_X] = 0;
  fields[FIELD_VOL_FLUX_Y] = 0;
  fields[FIELD_VOL_FLUX_Z] = 0;
  fields[FIELD_MASS_FLUX_X] = (direction==g_xdir);
  fields[FIELD_MASS_FLUX_Y] = (direction==g_ydir);
  fields[FIELD_MASS_FLUX_Z] = (direction==g_zdir);
  update_halo(fields,2);

  advec_mom(xvel, sweep_number, direction);
  advec_mom(yvel, sweep_number, direction);
  advec_mom(zvel, sweep_number, direction);

  sweep_number = 3;
  if(advect_x == TRUE) direction = g_zdir;
  if(!(advect_x == TRUE)) direction = g_xdir;

  advec_cell(sweep_number,direction);

  fields[FIELD_DENSITY0]  = 0;
  fields[FIELD_ENERGY0]   = 0;
  fields[FIELD_PRESSURE]  = 0;
  fields[FIELD_VISCOSITY] = 0;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_SOUNDSPEED] = 0;
  fields[FIELD_XVEL0]     = 0;
  fields[FIELD_YVEL0]     = 0;
  fields[FIELD_XVEL1]     = 1;
  fields[FIELD_YVEL1]     = 1;
  fields[FIELD_ZVEL0]     = 0;
  fields[FIELD_ZVEL1]     = 1;
  fields[FIELD_VOL_FLUX_X] = 0;
  fields[FIELD_VOL_FLUX_Y] = 0;
  fields[FIELD_VOL_FLUX_Z] = 0;
  fields[FIELD_MASS_FLUX_X] = (direction==g_xdir);
  fields[FIELD_MASS_FLUX_Y] = (direction==g_ydir);
  fields[FIELD_MASS_FLUX_Z] = (direction==g_zdir);
  update_halo(fields,2);

  advec_mom(xvel, sweep_number, direction);
  advec_mom(yvel, sweep_number, direction);
  advec_mom(zvel, sweep_number, direction);

}
