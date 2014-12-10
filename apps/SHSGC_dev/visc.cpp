/** @brief SHSGC top level program
  * @author Satya P. Jammy, converted to OPS by Gihan Mudalige
  * @details Including viscous terms in OPS -- Satya P Jammy
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
#include "xder12_kernel.h"

#include "flowvar_kernel.h"
#include "zerovisc_kernel.h"

void visc()
{
int nxp_range_1[] = {2,nxp-2};
int nxp_range[] = {0,nxp};



double ct = gam1 * gam * pow(Mach,2);
double cq = 1.0 / (gam1 * Pr * pow(Mach,2));


ops_par_loop(xder12_kernel, "xder12_kernel" ,shsgc_grid , 1 ,nxp_range_1,
			ops_arg_dat(u ,1 ,S1D_0M1M2P1P2 ,"double" ,OPS_READ),
			ops_arg_dat(u_x ,1 ,S1D_0 ,"double" ,OPS_WRITE),
			ops_arg_dat(u_xx ,1 ,S1D_0 ,"double" ,OPS_WRITE));


	// Evaluation of mu__x
ops_par_loop(xder1_kernel, "xder1_kernel" ,shsgc_grid , 1 ,nxp_range_1,
			ops_arg_dat(mu ,1 ,S1D_0M1M2P1P2, "double" ,OPS_READ),
			ops_arg_dat(mu_x ,1 ,S1D_0, "double" ,OPS_WRITE));
		

//
//Here is the calculation of viscous trems that involves derivatives
//

}
