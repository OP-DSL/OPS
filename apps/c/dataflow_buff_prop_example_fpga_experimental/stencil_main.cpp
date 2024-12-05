/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @Test application for fpga batched temporal blocked stencil 2D
  * @author Gihan Mudalige, Istvan Reguly, Beniel Thileepan
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

float dx,dy,dx_2,dy_2,dx_2_dy_2,dx_2_plus_dy_2_mult_2;
extern const unsigned short mem_vector_factor;

// OPS header file
#define OPS_2D
#define VERIFICATION
// #define OPS_CPP_API
#define OPS_HLS_V2
// #define OPS_FPGA
#define PROFILE
#include <ops_seq_v2.h>
#include "stencil_kernel.h"

#ifdef PROFILE
    #include <chrono>
#endif

/******************************************************************************
* Main program
*******************************************************************************/
int main(int argc, const char **argv)
{
  /**-------------------------- Initialisation --------------------------**/

    // OPS initialisation
    ops_init(argc,argv,1);


    //Mesh
    int imax = 20;
    int jmax = 20;
    unsigned int iter_max = 135;
    unsigned int batches = 1;

    const char* pch;
    for ( int n = 1; n < argc; n++ ) 
    {
        pch = strstr(argv[n], "-sizex=");

        if(pch != NULL) {
            imax = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-sizey=");

        if(pch != NULL) {
            jmax = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-iters=");

        if(pch != NULL) {
            iter_max = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-batch=");

        if(pch != NULL) {
            batches = atoi ( argv[n] + 7 ); continue;
        }
    }

    //The 2D block
    ops_block block;

    block = ops_decl_block(2, "block");

    //declare stencils
    int s2D_00[] = {0,0};
    ops_stencil S2D_00 = ops_decl_stencil(2, 1, s2D_00, "00");
    int s2D_00_P10_M10_0P1_0M1[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
    ops_stencil S2D_00_P10_M10_0P1_0M1 = ops_decl_stencil(2, 5, s2D_00_P10_M10_0P1_0M1, "00:10:-10:01:0-1");

    //declare datasets
    int size[] = {imax, jmax};
    int base[] = {0,0};
    int d_m[] = {-1,-1};
    int d_p[] = {1,1};
    float* temp = NULL;

    ops_dat dat0;
    ops_dat dat1;
    ops_dat dat2;
    ops_dat a;
    ops_dat b;

    // Allocation
    std::string name = std::string("dat0");
    dat0 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat1");
    dat1 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("dat2");
    dat2 = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("a");
    a = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());
    name = std::string("b");
    b = ops_decl_dat(block, 1, size, base, d_m, d_p, temp, "float", name.c_str());

    ops_partition("");

    int full_range[] = {d_m[0], size[0] + d_p[0], d_m[1], size[1] + d_p[1]};
    int internal_range[] = {0, size[0], 0, size[1]};


    //iterative stencil loop

    ops_printf("Launching stencil calculation: %d x %d mesh\n", size[0], size[1]);

#ifndef OPS_FPGA
    for (int iter = 0; iter < iter_max; iter++)
    {
        ops_par_loop(kernel_1_5pt, "stencil5pt", block, 2, internal_range,
                ops_arg_dat(a, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00, "float", OPS_WRITE));
        
        ops_par_loop(kernel_2_5pt, "stencil5pt", block, 2, internal_range,
                ops_arg_dat(b, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_WRITE));

        ops_par_loop(copy, "copy", block, 2, internal_range,
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE));
    }
#else

    ops_iter_par_loop("ops_iter_par_loop_0", iter_max,
        ops_par_loop(kernel_1_5pt, "stencil5pt", block, 2, internal_range,
                ops_arg_dat(a, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00, "float", OPS_WRITE)),
        ops_par_loop(kernel_2_5pt, "stencil5pt", block, 2, internal_range,
                ops_arg_dat(b, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat1, 1, S2D_00_P10_M10_0P1_0M1, "float", OPS_READ),
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_WRITE)),
        ops_par_loop(copy, "copy", block, 2, internal_range,
                ops_arg_dat(dat2, 1, S2D_00, "float", OPS_READ),
                ops_arg_dat(dat0, 1, S2D_00, "float", OPS_WRITE)));
#endif

    //cleaning
    ops_free_dat(dat0);
    ops_free_dat(dat1);
    ops_free_dat(dat2);
    ops_free_dat(a);
    ops_free_dat(b);

    ops_exit();

    std::cout << "Exit properly" << std::endl;
    return 0;
}
