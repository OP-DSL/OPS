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

/** @brief headder file declaring the functions for the ops sequential backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Declares the OPS API calls for the sequential backend
  */

#include "ops_lib_cpp.h"

inline void ops_arg_set(int n, ops_arg arg, char **p_arg){
        if (arg.stencil!=NULL) {
          for (int i = 0; i < arg.stencil->points; i++){
            p_arg[i] = arg.data + sizeof(double)*(n * arg.stencil->stride[0]  +
            arg.stencil->stencil[i*arg.stencil->dims + 0]);
          }
        } else {
          *p_arg = arg.data;
        }
}

inline void ops_args_set(int iter_x, int nargs, ops_arg *args, char ***p_a){
        for (int n=0; n<nargs; n++) {
          ops_arg_set(iter_x, args[n], p_a[n]);
        }
}



template < class T0 >
void ops_par_loop(void (*kernel)( T0* ),
  char const * name, int dim, int *range,
  ops_arg arg0 ) {

  char** p_a[1];
  ops_arg args[1] = {arg0};

  // consistency checks
  // ops_args_check(1,args,name);

  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
  }

  // loop over set elements

  if (dim == 1) {
      for (int n_x = range[0]; n_x < range[1]; n_x++) {
        ops_args_set(n_x, 1,args,p_a);
        // call kernel function, passing in pointers to data
        kernel( (T0 *)p_a[0] );
      }
    }


  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      free(p_a[i]);
  }
}
