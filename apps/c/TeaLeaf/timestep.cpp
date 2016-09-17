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
 TeaLeaf. if not, see http://www.gnu.org/licenses/. */

// @brief Controls the main diffusion cycle.
// @author Istvan Reguly, David Beckingsale, Wayne Gaudin
// @details Implicitly calculates the change in temperature using a Jacobi iteration


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include <ops_seq.h>


#include "data.h"
#include "definitions.h"

void timestep() {
	double dtlp;
	double kernel_time, c, t;

	if(profiler_on) ops_timers_core(&c,&kernel_time);


    //calc_dt(&dt)
    dt = dtinit;


  // if(profiler_on) profiler%timestep=profiler%timestep+(timer()-kernel_time)

    ops_fprintf(g_out, "Step %d time %g timestep %g\n", step, currtime, dt);
    ops_printf("Step %d time %g timestep %g\n", step, currtime, dt);
      

}

