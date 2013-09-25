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

/** @brief OPS core library functions
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS backends
  */

#include <sys/time.h>
#include "ops_lib_core.h"

int OPS_diags = 0;

int OPS_block_index = 0, OPS_block_max = 0;
int OPS_stencil_index = 0, OPS_stencil_max = 0;
int OPS_dat_index = 0;


/*
* Lists of blocks and dats declared in an OPS programs
*/

ops_block * OPS_block_list;
ops_stencil * OPS_stencil_list;
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

int compare_blocks(ops_block block1, ops_block block2)
{
  if(block1->dims == block2->dims && block1->index == block2->index &&
      strcmp(block1->name,block2->name)==0 )
    return 1;
  else return 0;
}

ops_dat search_dat(ops_block block, int size, int *block_size, int* offset,
  char const * type, char const * name)
{
  ops_dat_entry* item;
  ops_dat_entry* tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat item_dat = item->dat;

    if (strcmp(item_dat->name,name) == 0 && /* there are other components to compare*/
       (item_dat->size) == size && compare_blocks(item_dat->block, block) == 1 &&
       strcmp(item_dat->type,type) == 0 ) {
       return item_dat;
    }
  }

  return NULL;
}

/*
* OPS core functions
*/
void ops_init( int argc, char ** argv, int diags )
{
  OPS_diags = diags;

  /*Initialize the double linked list to hold ops_dats*/
  TAILQ_INIT(&OPS_dat_list);

}

void ops_exit( )
{
  // free storage and pointers for blocks
  for ( int i = 0; i < OPS_block_index; i++ ) {
    free((char*)OPS_block_list[i]->name);
    free((char*)OPS_block_list[i]->size);
    free(OPS_block_list[i]);
  }
  free(OPS_block_list);
  OPS_block_list = NULL;

  /*free doubl linked list holding the ops_dats */
  ops_dat_entry *item;
  while ((item = TAILQ_FIRST(&OPS_dat_list))) {
    if ((item->dat)->user_managed == 0)
      free((item->dat)->data);
    free((item->dat)->block_size);
    free((item->dat)->offset);
    free((char*)(item->dat)->name);
    free((char*)(item->dat)->type);
    TAILQ_REMOVE(&OPS_dat_list, item, entries);
    free(item);
  }

  // free stencills
  for ( int i = 0; i < OPS_block_index; i++ ) {
    free((char*)OPS_stencil_list[i]->name);
    free(OPS_stencil_list[i]->stencil);
    free(OPS_stencil_list[i]->stride);
    free(OPS_stencil_list[i]);
  }
  free(OPS_stencil_list);
  OPS_stencil_list = NULL;

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

  ops_block block = (ops_block)xmalloc(sizeof(ops_block_core));
  block->index = OPS_block_index;
  block->dims = dims;
  block->size =(int *)xmalloc(sizeof(int)*dims);
  memcpy(block->size,size,sizeof(int)*dims);
  block->name = copy_str(name);
  OPS_block_list[OPS_block_index++] = block;

  return block;
}

ops_dat ops_decl_dat_core( ops_block block, int data_size,
                      int *block_size, int* offset, char *data, int type_size,
                      char const * type,
                      char const * name )
{
  if ( block == NULL ) {
    printf ( "ops_decl_dat error -- invalid block for data: %s\n", name );
    exit ( -1 );
  }

  if ( data_size <= 0 ) {
    printf ( "ops_decl_dat error -- negative/zero number of items per grid point in data: %s\n", name );
    exit ( -1 );
  }

  ops_dat dat = ( ops_dat ) xmalloc ( sizeof ( ops_dat_core ) );
  dat->index = OPS_dat_index;
  dat->block = block;

  dat->size = type_size*data_size;

  dat->block_size =(int *)xmalloc(sizeof(int)*block->dims);
  memcpy(dat->block_size,block_size,sizeof(int)*block->dims);

  dat->offset =( int *)xmalloc(sizeof(int)*block->dims);
  memcpy(dat->offset,offset,sizeof(int)*block->dims);

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


ops_dat ops_decl_dat_temp_core ( ops_block block, int data_size,
  int *block_size, int* offset,  char * data, int type_size, char const * type, char const * name )
{
  //Check if this dat already exists in the double linked list
  ops_dat found_dat = search_dat(block, data_size, block_size, offset, type, name);
  if ( found_dat != NULL) {
    printf("ops_dat with name %s already exists, cannot create temporary ops_dat\n ", name);
    exit(2);
  }
  //if not found ...
  return ops_decl_dat_core ( block, data_size, block_size, offset, data, type_size, type, name );
}


ops_stencil ops_decl_stencil ( int dims, int points, int *sten, char const * name)
{

  if ( OPS_stencil_index == OPS_stencil_max ) {
    OPS_stencil_max += 10;
    OPS_stencil_list = (ops_stencil *) realloc(OPS_stencil_list,OPS_stencil_max * sizeof(ops_stencil));

    if ( OPS_stencil_list == NULL ) {
      printf ( " ops_decl_stencil error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_stencil stencil = (ops_stencil)xmalloc(sizeof(ops_stencil_core));
  stencil->index = OPS_stencil_index;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);;

  stencil->stencil = (int *)xmalloc(dims*points*sizeof(int));
  memcpy(stencil->stencil,sten,sizeof(int)*dims*points);

  stencil->stride = (int *)xmalloc(dims*sizeof(int));
  for (int i = 0; i < dims; i++) stencil->stride[i] = 1;
  OPS_stencil_list[OPS_stencil_index++] = stencil;

  return stencil;
}



ops_stencil ops_decl_strided_stencil ( int dims, int points, int *sten, int *stride, char const * name)
{

  if ( OPS_stencil_index == OPS_stencil_max ) {
    OPS_stencil_max += 10;
    OPS_stencil_list = (ops_stencil *) realloc(OPS_stencil_list,OPS_stencil_max * sizeof(ops_stencil));

    if ( OPS_stencil_list == NULL ) {
      printf ( " ops_decl_stencil error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_stencil stencil = (ops_stencil)xmalloc(sizeof(ops_stencil_core));
  stencil->index = OPS_stencil_index;
  stencil->points = points;
  stencil->dims = dims;
  stencil->name = copy_str(name);;

  stencil->stencil = (int *)xmalloc(dims*points*sizeof(int));
  memcpy(stencil->stencil,sten,sizeof(int)*dims*points);

  stencil->stride = (int *)xmalloc(dims*sizeof(int));
  memcpy(stencil->stride,stride,sizeof(int)*dims);

  OPS_stencil_list[OPS_stencil_index++] = stencil;

  return stencil;
}



ops_arg ops_arg_dat_core ( ops_dat dat, ops_stencil stencil, ops_access acc ) {
  ops_arg arg;
  arg.argtype = OPS_ARG_DAT;
  arg.dat = dat;
  arg.stencil = stencil;
  if ( dat != NULL ) {
    arg.data = dat->data;
  } else {
    arg.data = NULL;
  }
  arg.acc = acc;
  return arg;
}

ops_arg ops_arg_gbl_core ( char * data, int dim, int size, ops_access acc ) {
  ops_arg arg;
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = data;
  arg.acc = acc;
  return arg;
}


void ops_diagnostic_output ( )
{
  if ( OPS_diags > 1 ) {
    printf ( "\n OPS diagnostic output\n" );
    printf ( " --------------------\n" );

    printf ( "\n block dimension [dims]\n" );
    printf ( " -------------------\n" );
    for ( int n = 0; n < OPS_block_index; n++ ) {
      printf ( " %15s %15dD ", OPS_block_list[n]->name, OPS_block_list[n]->dims );
      for (int i=0; i<OPS_block_list[n]->dims; i++)
        printf ( "[%d]",OPS_block_list[n]->size[i] );
      printf("\n");
    }

    printf ( "\n dats item/point [block_size] [offset]  block\n" );
    printf ( " ------------------------------\n" );
    ops_dat_entry *item;
    TAILQ_FOREACH(item, &OPS_dat_list, entries) {
      printf ( " %15s %15d ", (item->dat)->name, (item->dat)->size );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->block_size[i] );
      printf ( " " );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->offset[i] );

      printf ( " %15s\n", (item->dat)->block->name );
    }
    printf ( "\n" );
  }
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


void ops_print_dat_to_txtfile_core(ops_dat dat, const char* file_name)
{
  FILE *fp;
  if ( (fp = fopen(file_name,"a")) == NULL) {
    printf("can't open file %s\n",file_name);
    exit(2);
  }

  if (fprintf(fp,"ops_dat:  %s \n", dat->name)<0) {
    printf("error writing to %s\n",file_name);
    exit(2);
  }
  if (fprintf(fp,"Dims : %d ", dat->block->dims)<0) {
      printf("error writing to %s\n",file_name);
      exit(2);
    }

  for(int i = 0; i < dat->block->dims; i++) {
    if (fprintf(fp,"[%d]", dat->block_size[i])<0) {
      printf("error writing to %s\n",file_name);
      exit(2);
    }
  }
  fprintf(fp,"\n");

  if (fprintf(fp,"size %d \n", dat->size)<0) {
    printf("error writing to %s\n",file_name);
    exit(2);
  }

  if(dat->block->dims == 2) {
    if( strcmp(dat->type,"double") == 0 ) {
      for(int i = 0; i < dat->block_size[1]; i++ ) {
        for(int j = 0; j < dat->block_size[0]; j++ ) {
          if (fprintf(fp, "%3.10lf ",
            ((double *)dat->data)[i*dat->block_size[0]+j])<0) {
            printf("error writing to %s\n",file_name);
            exit(2);
          }
        }
        fprintf(fp,"\n");
      }
    }
    else if( strcmp(dat->type,"float") == 0 ) {
      for(int i = 0; i < dat->block_size[1]; i++ ) {
        for(int j = 0; j < dat->block_size[0]; j++ ) {
          if (fprintf(fp, "%e ", ((float *)dat->data)[i*dat->block_size[0]+j])<0) {
            printf("error writing to %s\n",file_name);
            exit(2);
          }
        }
        fprintf(fp,"\n");
      }
    }
    else if( strcmp(dat->type,"int") == 0 ) {
      for(int i = 0; i < dat->block_size[1]; i++ ) {
        for(int j = 0; j < dat->block_size[0]; j++ ) {
          if (fprintf(fp, "%d ", ((int *)dat->data)[i*dat->block_size[0]+j])<0) {
            printf("error writing to %s\n",file_name);
            exit(2);
          }
        }
        fprintf(fp,"\n");
      }
    }
    else {
      printf("Unknown type %s, cannot be written to file %s\n",dat->type,file_name);
      exit(2);
    }
    fprintf(fp,"\n");
  }
  else if(dat->block->dims == 1) {
    if( strcmp(dat->type,"double") == 0 ) {
      for(int j = 0; j < dat->block_size[0]; j++ ) {
        if (fprintf(fp, "%3.10lf ", ((double *)dat->data)[j])<0) {
          printf("error writing to %s\n",file_name);
          exit(2);
        }
      }
      fprintf(fp,"\n");
    }
    else if( strcmp(dat->type,"float") == 0 ) {
      for(int j = 0; j < dat->block_size[0]; j++ ) {
        if (fprintf(fp, "%e ", ((float *)dat->data)[j])<0) {
          printf("error writing to %s\n",file_name);
          exit(2);
        }
      }
      fprintf(fp,"\n");
    }
    else if( strcmp(dat->type,"int") == 0 ) {
      for(int j = 0; j < dat->block_size[0]; j++ ) {
        if (fprintf(fp, "%d ", ((int *)dat->data)[j])<0) {
          printf("error writing to %s\n",file_name);
          exit(2);
        }
      }
      fprintf(fp,"\n");
    }
    else {
      printf("Unknown type %s, cannot be written to file %s\n",dat->type,file_name);
      exit(2);
    }
    fprintf(fp,"\n");
  }
  fclose(fp);
}
