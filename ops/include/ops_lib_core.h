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

/** @brief ops core library function declarations
  * @author Gihan Mudalige
  * @details function declarations headder file for the core library functions
  * utilized by all OPS backends
  */

#ifndef __OPS_LIB_CORE_H
#define __OPS_LIB_CORE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdarg.h>
#include <sys/queue.h> //contains double linked list implementation
#include <stdbool.h>

#include "ops_util.h"

/*
* essential typedefs
*/

typedef unsigned int uint;
typedef long long ll;
typedef unsigned long long ull;


/*
* enum list for ops_par_loop
*/

#define OPS_READ 0
#define OPS_WRITE 1
#define OPS_RW 2
#define OPS_INC 3
#define OPS_MIN 4
#define OPS_MAX 5

#define OPS_ARG_GBL 0
#define OPS_ARG_DAT 1

typedef int ops_access; //holds OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX
typedef int ops_arg_type; // holds OP_ARG_GBL, OP_ARG_DAT


/*
* structures
*/

typedef struct
{
  int         index;  /* index */
  int         dims;   /* dimension of block, 2D,3D .. etc*/
  int         *size;  /* size of block in each dimension */
  char const  *name;  /* name of block */
} ops_block_core;

typedef ops_block_core * ops_block;

typedef struct
{
  int         index;       /* index */
  ops_block   block;       /* block on which data is defined */
  int         size;        /* number of data items per grid point*/
  int         *block_size; /* size of the array in each block dimension*/
  int         *offset;     /* starting index for each dimention*/
  char        *data;       /* data on host */
  char const  *name;       /* name of dataset */
  char const *type;        /* datatype */
  int         user_managed;/* indicates whether the user is managing memory */
} ops_dat_core;

typedef ops_dat_core * ops_dat;

//struct definition for a double linked list entry to hold an ops_dat
struct ops_dat_entry_core{
  ops_dat dat;
  TAILQ_ENTRY(ops_dat_entry_core) entries; /*holds pointers to next and
                                             previous entries in the list*/
};

typedef struct ops_dat_entry_core ops_dat_entry;

typedef TAILQ_HEAD(, ops_dat_entry_core) Double_linked_list;

typedef struct
{
        int         index;     /* index */
        int         dims;      /* dimensionality of the stencil */
        int         points;    /* number of stencil elements */
        char const  *name;     /* name of pointer */
        int         *stencil;  /* elements in the stencil */
        int         *stride;   /* stride of the stencil */
} ops_stencil_core;

typedef ops_stencil_core * ops_stencil;

typedef struct
{
  ops_dat     dat;     /* dataset */
  ops_stencil stencil; /* the stencil */
  int         dim;     /* dimension of data */
  char        *data;   /* data on host */
  ops_access   acc;    /* access type */
  ops_arg_type argtype;/* arg type */
} ops_arg;


/*******************************************************************************
* Core lib function prototypes
*******************************************************************************/

void ops_init( int argc, char **argv, int diags_level );

void ops_exit( void );

ops_block ops_decl_block(int dims, int *size, char *name);

ops_dat ops_decl_dat_core( ops_block block, int data_size,
                      int *block_size, int* offset, char *data, int type_size,
                      char const * type,
                      char const * name );

ops_dat ops_decl_dat_temp_core( ops_block block, int data_size,
                      int *block_size, int* offset,  char * data, int type_size,
                      char const * type, char const * name );

ops_stencil ops_decl_stencil( int dims, int points, int *stencil, char const * name);
ops_stencil ops_decl_strided_stencil( int dims, int points, int *sten, int *stride, char const * name);

ops_arg ops_arg_dat_core( ops_dat dat, ops_stencil stencil, ops_access acc );

void ops_printf(const char* format, ...);
void ops_fprintf(FILE *stream, const char *format, ...);

void ops_diagnostic_output( );


#endif /* __OP_LIB_CORE_H */
