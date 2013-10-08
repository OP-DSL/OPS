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

inline int ops_offs_set(int n_x,
                        int n_y, ops_arg arg){

        return
        arg.dat->block_size[0] * //multiply by the number of
        (n_y - arg.dat->offset[1])  // calculate the offset from index 0 for y dim
        +
        (n_x - arg.dat->offset[0]); //calculate the offset from index 0 for x dim


}


inline void ops_arg_set_opt(int n_x,
                        int n_y, ops_arg arg, char **p_arg){
  if (arg.stencil!=NULL) {
    for (int i = 0; i < arg.stencil->points; i++){
      p_arg[i] =
        arg.data + //base of 2D array
        //y dimension -- get to the correct y line
        arg.dat->size * arg.dat->block_size[0] * ( //multiply by the number of
                                                    //bytes per element and xdim block size
        (n_y - arg.dat->offset[1])  // calculate the offset from index 0 for y dim
        + // jump in strides in y dim ??
        arg.stencil->stencil[i*arg.stencil->dims + 1]) //get the value at the ith
                                                       //stencil point "+ 1" is the y dim
        +
        //x dimension - get to the correct x point on the y line
        arg.dat->size * ( //multiply by the number of bytes per element
        (n_x - arg.dat->offset[0])  //calculate the offset from index 0 for x dim
        + // jump in strides in x dim ??
        arg.stencil->stencil[i*arg.stencil->dims + 0] //get the value at the ith
                                                      //stencil point "+ 0" is the x dim
      );
    }
  } else {
    *p_arg = arg.data;
  }
}


inline void ops_args_set_opt(int iter_x,
                         int iter_y,
                         int nargs, ops_arg *args, char ***p_a){
  for (int n=0; n<nargs; n++) {
    ops_arg_set_opt(iter_x, iter_y, args[n], p_a[n]);
  }
}

template < class T0>
void ops_par_loop_opt(void (*kernel)( T0*),
                  char const * name, int dim, int *range,
                  ops_arg arg0) {

  char  **p_a[1];
  int   offs[1][2];
  int   count[dim];

  ops_arg args[1] = {arg0};

  for(int i=0; i<1; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

    if (args[i].stencil->stride[0] == 0) //stride in y as x stride is 0
    {
      offs[i][0] = 0;
      offs[i][1] = args[i].dat->block_size[0];
    }
    else if (args[i].stencil->stride[1] == 0) //stride in x as y stride is 0
    {
      offs[i][0] = 1;
      offs[i][1] = -( (range[1] - args[i].dat->offset[0]) -
                      (range[0] - args[i].dat->offset[0]) ) +1;
    }
    //printf("offs[i][0] = %d,  offs[i][1] = %d\n", offs[i][0], offs[i][1]);
  }

  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],1,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<1; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]) ;
      }
    }
  }

  for (int i = 0; i < 1; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}



template < class T0, class T1 >
void ops_par_loop_opt(void (*kernel)( T0*, T1* ),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1 ) {

  char  **p_a[2];
  int   offs[2][2];
  int   count[dim];
  ops_arg args[2] = {arg0, arg1};

  for(int i=0; i<2; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }


  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],2,args,p_a); //set up the initial possition

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<2; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 2; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}




template < class T0, class T1, class T2>
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2 ) {

  char  **p_a[3];
  int   offs[3][2];
  int   count[dim];

  ops_arg args[3] = {arg0, arg1, arg2};

  for(int i=0; i<3; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],3,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<3; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}


template < class T0, class T1, class T2, class T3  >
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3  ) {

  char  **p_a[4];
  int   offs[4][2];
  int   count[dim];

  ops_arg args[4] = {arg0, arg1, arg2, arg3};

  for(int i=0; i<4; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 4; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],4,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] );

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<4; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}


//5 args
template < class T0, class T1, class T2, class T3 , class T4>
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
                  ops_arg arg4) {

  char  **p_a[5];
  int   offs[5][2];
  int   count[dim];

  ops_arg args[5] = {arg0, arg1, arg2, arg3, arg4};

  for(int i=0; i<5; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],5,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<5; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}

//5 args
template < class T0, class T1, class T2, class T3 , class T4>
void ops_par_loop_opt2(void (*kernel)( T0*, T1*, T2*, T3*, T4*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
                  ops_arg arg4) {

  char  **p_a[5];
  int   offs[5][2];
  int   count[dim];

  ops_arg args[5] = {arg0, arg1, arg2, arg3, arg4};

  for(int i=0; i<5; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

    if (args[i].stencil->stride[0] == 0) //stride in y as x stride is 0
    {
      offs[i][0] = 0;
      offs[i][1] = args[i].dat->block_size[0];
    }
    else if (args[i].stencil->stride[1] == 0) //stride in x as y stride is 0
    {
      offs[i][0] = 1;
      offs[i][1] = -( range[1] - range[0] ) +1;
    }
    //printf("offs[i][0] = %d,  offs[i][1] = %d\n", offs[i][0], offs[i][1]);
  }

  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],5,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<5; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 5; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}




//6 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5>
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2,
                  ops_arg arg3, ops_arg arg4, ops_arg arg5) {

  char  **p_a[6];
  int   offs[6][2];
  int   count[dim];

  ops_arg args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};

  for(int i=0; i<6; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],6,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<6; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);
    }
  }
}



//6 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5>
void ops_par_loop_opt2(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2,
                  ops_arg arg3, ops_arg arg4, ops_arg arg5) {

  char  **p_a[6];
  int   offs[6][2];
  int   count[dim];

  ops_arg args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};

  for(int i=0; i<6; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

    if (args[i].stencil->stride[0] == 0) //stride in y as x stride is 0
    {
      offs[i][0] = 0;
      offs[i][1] = args[i].dat->block_size[0];
    }
    else if (args[i].stencil->stride[1] == 0) //stride in x as y stride is 0
    {
      offs[i][0] = 1;
      offs[i][1] = -( range[1] - range[0] ) +1;
    }
    //printf("offs[i][0] = %d,  offs[i][1] = %d\n", offs[i][0], offs[i][1]);
  }

  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],6,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<6; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);
    }
  }
}



//7 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6>
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2,
                  ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6) {

  char  **p_a[7];
  int   offs[7][2];
  int   count[dim];

  ops_arg args[7] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6};

  for(int i=0; i<7; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],7,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<7; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]) ;
      }
    }
  }

  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);
    }
  }
}


//7 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6>
void ops_par_loop_opt2(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2,
                  ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6) {

  char  **p_a[7];
  int   offs[7][2];
  int   count[dim];

  ops_arg args[7] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6};

  for(int i=0; i<7; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

    if (args[i].stencil->stride[0] == 0) //stride in y as x stride is 0
    {
      offs[i][0] = 0;
      offs[i][1] = args[i].dat->block_size[0];
    }
    else if (args[i].stencil->stride[1] == 0) //stride in x as y stride is 0
    {
      offs[i][0] = 1;
      offs[i][1] = -( (range[1] - args[i].dat->offset[0]) -
                      (range[0] - args[i].dat->offset[0]) ) +1;
    }
  }

  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],7,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<7; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]) ;
      }
    }
  }

  for (int i = 0; i < 7; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);
    }
  }
}


//8 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7>
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  char  **p_a[8];
  int   offs[8][2];
  int   count[dim];

  ops_arg args[8] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7};

  for(int i=0; i<8; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],8,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<8; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}



//8 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7>
void ops_par_loop_opt2(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7) {

  char  **p_a[8];
  int   offs[8][2];
  int   count[dim];

  ops_arg args[8] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7};

  for(int i=0; i<8; i++) {
    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;

    if (args[i].stencil->stride[0] == 0) //stride in y as x stride is 0
    {
      offs[i][0] = 0;
      offs[i][1] = args[i].dat->block_size[0];
    }
    else if (args[i].stencil->stride[1] == 0) //stride in x as y stride is 0
    {
      offs[i][0] = 1;
      offs[i][1] = -( range[1] - range[0] ) +1;
    }
    //printf("offs[i][0] = %d,  offs[i][1] = %d\n", offs[i][0], offs[i][1]);
  }

  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],8,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<8; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 8; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}

//11 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7, class T8 , class T9,
           class T10 >
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*, T8*, T9*,
                                      T10*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7, ops_arg arg8, ops_arg arg9,
                  ops_arg arg10) {

  char  **p_a[11];
  int   offs[11][2];
  int   count[dim];

  ops_arg args[11] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7, arg8, arg9,
                      arg10};

  for(int i=0; i<11; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 11; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],11,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8] , (T9 *)p_a[9],
            (T10 *)p_a[10]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<11; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 11; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}



//12 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7, class T8 , class T9,
           class T10, class T11 >
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*, T8*, T9*,
                                      T10*, T11*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7, ops_arg arg8, ops_arg arg9,
                  ops_arg arg10, ops_arg arg11) {

  char  **p_a[12];
  int   offs[12][2];
  int   count[dim];

  ops_arg args[12] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7, arg8, arg9,
                      arg10, arg11};

  for(int i=0; i<12; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 12; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],12,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8] , (T9 *)p_a[9],
            (T10 *)p_a[10], (T11 *)p_a[11]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<12; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 12; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}



//14 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7, class T8 , class T9,
           class T10, class T11, class T12, class T13 >
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*, T8*, T9*,
                                      T10*, T11*, T12*, T13*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7, ops_arg arg8, ops_arg arg9,
                  ops_arg arg10, ops_arg arg11, ops_arg arg12, ops_arg arg13) {

  char  **p_a[14];
  int   offs[14][2];
  int   count[dim];

  ops_arg args[14] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7, arg8, arg9,
                      arg10, arg11, arg12, arg13};

  for(int i=0; i<14; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 14; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],14,args,p_a);

  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8] , (T9 *)p_a[9],
            (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<14; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 14; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}


//20 args
template < class T0, class T1, class T2, class T3 , class T4,
           class T5, class T6, class T7, class T8 , class T9,
           class T10, class T11, class T12, class T13 , class T14,
           class T15, class T16, class T17, class T18 , class T19 >
void ops_par_loop_opt(void (*kernel)( T0*, T1*, T2*, T3*, T4*,
                                      T5*, T6*, T7*, T8*, T9*,
                                      T10*, T11*, T12*, T13*, T14*,
                                      T15*, T16*, T17*, T18*, T19*),
                  char const * name, int dim, int *range,
                  ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3, ops_arg arg4,
                  ops_arg arg5, ops_arg arg6, ops_arg arg7, ops_arg arg8, ops_arg arg9,
                  ops_arg arg10, ops_arg arg11, ops_arg arg12, ops_arg arg13, ops_arg arg14,
                  ops_arg arg15, ops_arg arg16, ops_arg arg17, ops_arg arg18, ops_arg arg19) {

  char  **p_a[20];
  int   offs[20][2];
  int   count[dim];

  ops_arg args[20] = {arg0, arg1, arg2, arg3, arg4,
                      arg5, arg6, arg7, arg8, arg9,
                      arg10, arg11, arg12, arg13, arg14,
                      arg15, arg16, arg17, arg19, arg19};

  for(int i=0; i<20; i++) {

    offs[i][0] = 1;  //unit step in x dimension
    offs[i][1] = ops_offs_set(range[0],range[2]+1, args[i]) - ops_offs_set(range[1],range[2], args[i]) +1;
  }

  for (int i = 0; i < 20; i++) {
    if (args[i].argtype == OPS_ARG_DAT)
      p_a[i] = (char **)malloc(args[i].stencil->points * sizeof(char *));
    else if (args[i].argtype == OPS_ARG_GBL)
      p_a[i] = (char **)malloc(args[i].dim * sizeof(char *));
  }

  int total_range = 1;
  for (int m=0; m<dim; m++) {
    count[m] = range[2*m+1]-range[2*m];  // number in each dimension
    total_range *= count[m];
  }
  count[dim-1]++;     // extra in last to ensure correct termination

  //set up initial pointers
  ops_args_set(range[0], range[2],20,args,p_a);


  for (int nt=0; nt<total_range; nt++) {

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_a[0], (T1 *)p_a[1], (T2 *)p_a[2], (T3 *)p_a[3] , (T4 *)p_a[4],
            (T5 *)p_a[5], (T6 *)p_a[6], (T7 *)p_a[7], (T8 *)p_a[8] , (T9 *)p_a[9],
            (T10 *)p_a[10], (T11 *)p_a[11], (T12 *)p_a[12], (T13 *)p_a[13] , (T14 *)p_a[14],
            (T15 *)p_a[15], (T16 *)p_a[16], (T17 *)p_a[17], (T18 *)p_a[18] , (T19 *)p_a[19]);

    count[0]--;   // decrement counter
    int m = 0;    // max dimension with changed index

    while (count[m]==0) {
      count[m] = range[2*m+1]-range[2*m]; // reset counter
      m++;                                // next dimension
      count[m]--;                         // decrement counter
    }

    // shift pointers to data
    for (int i=0; i<20; i++) {
      for (int np=0; np<args[i].stencil->points; np++) {
        p_a[i][np] = p_a[i][np] + (args[i].dat->size * offs[i][m]);
      }
    }
  }

  for (int i = 0; i < 20; i++) {
    if (args[i].argtype == OPS_ARG_DAT) {
      free(p_a[i]);

    }
  }
}
