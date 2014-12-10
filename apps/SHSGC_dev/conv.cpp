/** @brief SHSGC top level program
  * @author Satya P. Jammy
  * @details 
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing multi-block Structured mesh applications
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "vars.h"
#include "ops_data.h"


#include "xder1_kernel.h"
#include "drhouupdx_kernel.h"
#include "drhoEpudx_kernel.h"


void conv()
{
	int nxp_range_1[] = {2,nxp-2};
	int nxp_range[] = {0,nxp};
	
ops_par_loop(xder1_kernel, "xder1_kernel", shsgc_grid, 1, nxp_range_1,
            ops_arg_dat(rhou_new, 1, S1D_0M1M2P1P2, "double",OPS_READ),
            ops_arg_dat(rho_res, 1, S1D_0, "double",OPS_WRITE));

// calculate (rhouu + p)
ops_par_loop(drhouupdx_kernel, "drhouupdx_kernel", shsgc_grid, 1, nxp_range,
            ops_arg_dat(rhou_new, 1, S1D_0, "double",OPS_READ),
            ops_arg_dat(rho_new,  1, S1D_0, "double",OPS_READ),
            ops_arg_dat(rhoE_new, 1, S1D_0, "double",OPS_READ),
            ops_arg_dat(fn,  1, S1D_0, "double",OPS_WRITE));
	  
// calculate derivative
	 
ops_par_loop(xder1_kernel, "xder1_kernel", shsgc_grid, 1, nxp_range_1,
            ops_arg_dat(fn, 1, S1D_0M1M2P1P2, "double",OPS_READ),
            ops_arg_dat(rhou_res,  1, S1D_0, "double",OPS_WRITE));

// Energy equation cal (rhoE+p)u
	  
ops_par_loop(drhoEpudx_kernel, "drhoEpudx_kernel", shsgc_grid, 1, nxp_range,
            ops_arg_dat(rhou_new, 1, S1D_0, "double",OPS_READ),
            ops_arg_dat(rho_new,  1, S1D_0, "double",OPS_READ),
            ops_arg_dat(rhoE_new, 1, S1D_0, "double",OPS_READ),
            ops_arg_dat(fn,  1, S1D_0, "double",OPS_WRITE));
// cal derivative 
ops_par_loop(xder1_kernel, "xder1_kernel", shsgc_grid, 1, nxp_range_1,
            ops_arg_dat(fn, 1, S1D_0M1M2P1P2, "double",OPS_READ),
            ops_arg_dat(rhoE_res,  1, S1D_0, "double",OPS_WRITE));
	
}

