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

#include <sys/time.h>
#include "ops_lib_core.h"

int OP_diags = 0;

int OPS_block_index = 0, OPS_block_max = 0;
int OPS_dat_index = 0;


/*
* Lists of blocks and dats declared in an OPS programs
*/

ops_block * OPS_block_list;
Double_linked_list OPS_dat_list; //Head of the double linked list


/*
* Utility functions
*/
static char * copy_str( char const * src )
{
  const size_t len = strlen( src ) + 1;
  char * dest = (char *) calloc ( len, sizeof ( char ) );
  return strncpy ( dest, src, len );
}

/*
* OPS core functions
*/
void ops_init( int argc, char ** argv, int diags )
{
  OP_diags = diags;

  /*Initialize the double linked list to hold ops_dats*/
  TAILQ_INIT(&OPS_dat_list);

}

void ops_exit( )
{
  // free storage and pointers for blocks
  for ( int i = 0; i < OPS_block_index; i++ ) {
    free((char*)OPS_block_list[i]->name);
    free(OPS_block_list[i]);
  }
  free(OPS_block_list);
  OPS_block_list = NULL;

  /*free doubl linked list holding the ops_dats */
  ops_dat_entry *item;
  while ((item = TAILQ_FIRST(&OPS_dat_list))) {
    if (!(item->dat)->user_managed)
      free((item->dat)->data);
    free((char*)(item->dat)->name);
    free((char*)(item->dat)->type);
    TAILQ_REMOVE(&OPS_dat_list, item, entries);
    free(item);
  }

  // reset initial values
  OPS_block_index = 0;
  OPS_dat_index = 0;
  OPS_block_max = 0;
}

ops_block ops_decl_block(int dims, int *size, char *name)
{
  if ( dims < 0 ) {
    printf ( " ops_decl_block error -- negative/zero dimension size for block: %s\n", name );
    exit ( -1 );
  }

  if ( OPS_block_index == OPS_block_max ) {
    OPS_block_max += 10;
    OPS_block_list = (ops_block *) realloc(OPS_block_list,OPS_block_max * sizeof(ops_block));

    if ( OPS_block_list == NULL ) {
      printf ( " ops_decl_block error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_block block = (ops_block)malloc(sizeof(ops_block_core));
  block->index = OPS_block_index;
  block->dims = dims;
  block->size = size;
  block->name = copy_str(name);
  OPS_block_list[OPS_block_index++] = block;

  return block;
}

ops_dat ops_decl_dat_core( ops_block block, int data_size,
                      int *block_size, int* offset, char *data,
                      char const * type,
                      char const * name )
{
  if ( block == NULL )
  {
    printf ( "ops_decl_dat error -- invalid block for data: %s\n", name );
    exit ( -1 );
  }

  if ( data_size <= 0 )
  {
    printf ( "ops_decl_dat error -- negative/zero number of items per grid point in data: %s\n", name );
    exit ( -1 );
  }

  ops_dat dat = ( ops_dat ) malloc ( sizeof ( ops_dat_core ) );
  dat->index = OPS_dat_index;
  dat->block = block;
  dat->data_size = data_size;
  dat->data = (char *)data;
  dat->user_managed = 1;
  dat->type = copy_str( type );
  dat->name = copy_str(name);

  /* Create a pointer to an item in the ops_dats doubly linked list */
  ops_dat_entry* item;

  //add the newly created ops_dat to list
  item = (ops_dat_entry *)malloc(sizeof(ops_dat_entry));
  if (item == NULL) {
    printf ( " op_decl_dat error -- error allocating memory to double linked list entry\n" );
    exit ( -1 );
  }
  item->dat = dat;
  Double_linked_list test; //Head of the double linked list
  //add item to the end of the list
  TAILQ_INSERT_TAIL(&OPS_dat_list, item, entries);
  OPS_dat_index++;

  return dat;
}



void ops_printf(const char* format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void ops_fprintf(FILE *stream, const char *format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stream, format, argptr);
  va_end(argptr);
}
