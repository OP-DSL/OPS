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
#include <sys/time.h>
#include <float.h>
#include <limits.h>

int OPS_diags = 0;

int OPS_block_index = 0, OPS_block_max = 0;
int OPS_stencil_index = 0, OPS_stencil_max = 0;
int OPS_dat_index = 0;
int OPS_kern_max=0, OPS_kern_curr=0;
ops_kernel * OPS_kernels=NULL;
ops_arg *OPS_curr_args=NULL;
const char *OPS_curr_name=NULL;
int OPS_hybrid_gpu = 0, OPS_gpu_direct = 0;
int OPS_halo_group_index = 0, OPS_halo_group_max = 0,
    OPS_halo_index = 0, OPS_halo_max = 0,
    OPS_reduction_index = 0, OPS_reduction_max = 0;
ops_reduction * OPS_reduction_list = NULL;
int OPS_enable_checkpointing = 0;

/*
* Lists of blocks and dats declared in an OPS programs
*/

ops_block_descriptor * OPS_block_list = NULL;
ops_halo * OPS_halo_list = NULL;
ops_halo_group * OPS_halo_group_list = NULL;
ops_stencil * OPS_stencil_list = NULL;
Double_linked_list OPS_dat_list; //Head of the double linked list

int OPS_block_size_x = 32;
int OPS_block_size_y = 4;


double ops_gather_time=0.0;
double ops_scatter_time=0.0;
double ops_sendrecv_time=0.0;
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

ops_dat search_dat(ops_block block, int elem_size, int *dat_size, int* offset,
  char const * type, char const * name)
{
  ops_dat_entry* item;
  ops_dat_entry* tmp_item;
  for (item = TAILQ_FIRST(&OPS_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    ops_dat item_dat = item->dat;

    if (strcmp(item_dat->name,name) == 0 && /* there are other components to compare*/
       (item_dat->elem_size) == elem_size && compare_blocks(item_dat->block, block) == 1 &&
       strcmp(item_dat->type,type) == 0 ) {
       return item_dat;
    }
  }

  return NULL;
}

/*
* OPS core functions
*/
void ops_init_core( int argc, char ** argv, int diags )
{
  OPS_diags = diags;

  for ( int n = 1; n < argc; n++ )
  {
    if ( strncmp ( argv[n], "OPS_BLOCK_SIZE_X=", 17 ) == 0 )
    {
      OPS_block_size_x = atoi ( argv[n] + 17 );
      printf ( "\n OPS_block_size_x = %d \n", OPS_block_size_x );
    }

    if ( strncmp ( argv[n], "OPS_BLOCK_SIZE_Y=", 17 ) == 0 )
    {
      OPS_block_size_y = atoi ( argv[n] + 17 );
      printf ( "\n OPS_block_size_y = %d \n", OPS_block_size_y );
    }

    if ( strncmp ( argv[n], "-gpudirect", 10 ) == 0 )
    {
      OPS_gpu_direct = 1;
      ops_printf ( "\n GPU Direct enabled\n" );
    }
    if ( strncmp ( argv[n], "OPS_DIAGS=", 10 ) == 0 )
    {
      OPS_diags = atoi ( argv[n] + 10 );
      printf ( "\n OPS_diags = %d \n", OPS_diags );
    }
    if ( strncmp ( argv[n], "OPS_CHECKPOINT", 14 ) == 0 )
    {
      OPS_enable_checkpointing = 1;
      printf ( "\n OPS Checkpointing enabled\n");
    }
  }

  /*Initialize the double linked list to hold ops_dats*/
  TAILQ_INIT(&OPS_dat_list);

}

void ops_exit_core( )
{
  ops_dat_entry *item;
  // free storage and pointers for blocks
  for ( int i = 0; i < OPS_block_index; i++ ) {
    free((char*)(OPS_block_list[i].block->name));
    while ((item = TAILQ_FIRST(&(OPS_block_list[i].datasets)))) {
      TAILQ_REMOVE(&(OPS_block_list[i].datasets), item, entries);
      free(item);
    }
    free(OPS_block_list[i].block);
  }
  free(OPS_block_list);
  OPS_block_list = NULL;

  /*free doubly linked list holding the ops_dats */

  while ((item = TAILQ_FIRST(&OPS_dat_list))) {
    if ((item->dat)->user_managed == 0)
      free((item->dat)->data);
    free((char*)(item->dat)->name);
    free((char*)(item->dat)->type);
    TAILQ_REMOVE(&OPS_dat_list, item, entries);
    free(item->dat);
    free(item);
  }

  // free stencills
  for ( int i = 0; i < OPS_stencil_index; i++ ) {
    free((char*)OPS_stencil_list[i]->name);
    free(OPS_stencil_list[i]->stencil);
    free(OPS_stencil_list[i]->stride);
    free(OPS_stencil_list[i]);
  }
  free(OPS_stencil_list);
  OPS_stencil_list = NULL;

  for (int i = 0; i < OPS_halo_index; i++) {
    free(OPS_halo_list[i]);
  }

  for (int i = 0; i < OPS_halo_group_index; i++) {
    //free(OPS_halo_group_list[i]->halos); //TODO: we didn't make a copy
    free(OPS_halo_group_list[i]);
  }

  // reset initial values
  OPS_block_index = 0;
  OPS_dat_index = 0;
  OPS_block_max = 0;

  ops_checkpointing_exit();
}

ops_block ops_decl_block(int dims, char *name)
{
  if ( dims < 0 ) {
    printf ( " ops_decl_block error -- negative/zero dimension size for block: %s\n", name );
    exit ( -1 );
  }

  if ( OPS_block_index == OPS_block_max ) {
    OPS_block_max += 10;
    OPS_block_list = (ops_block_descriptor *) realloc(OPS_block_list,OPS_block_max * sizeof(ops_block_descriptor));

    if ( OPS_block_list == NULL ) {
      printf ( " ops_decl_block error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_block block = (ops_block)xmalloc(sizeof(ops_block_core));
  block->index = OPS_block_index;
  block->dims = dims;
  block->name = copy_str(name);
  OPS_block_list[OPS_block_index].block = block;
  OPS_block_list[OPS_block_index].num_datasets = 0;
  TAILQ_INIT(&(OPS_block_list[OPS_block_index].datasets));
  OPS_block_index++;

  return block;
}

void ops_decl_const_core( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

ops_dat ops_decl_dat_core( ops_block block, int dim,
                      int *dataset_size, int* base, int* d_m, int* d_p, char *data, int type_size,
                      char const * type,
                      char const * name )
{
  if ( block == NULL ) {
    printf ( "ops_decl_dat error -- invalid block for data: %s\n", name );
    exit ( -1 );
  }

  if ( dim <= 0 ) {
    printf ( "ops_decl_dat error -- negative/zero number of items per grid point in data: %s\n", name );
    exit ( -1 );
  }

  ops_dat dat = ( ops_dat ) xmalloc ( sizeof ( ops_dat_core ) );
  dat->index = OPS_dat_index;
  dat->block = block;
  dat->dim = dim;
  dat->elem_size = type_size*dim;  //note here that the element size is taken to
                                   //be the type_size in bytes multiplied by the dimension of an element
  dat->e_dat = 0; //default to non-edge dat

  for(int n=0;n<block->dims;n++){
    if(dataset_size[n] != 1) {
      //compute total size - which includes the block halo
      dat->size[n] = dataset_size[n] - d_m[n] + d_p[n];
    }
    else {
      dat->size[n] = 1;
      dat->e_dat = 1;
    }
  }

  for(int n=0;n<block->dims;n++) dat->base[n] = base[n];

  for(int n=0;n<block->dims;n++) dat->d_m[n] = d_m[n];
  for(int n=0;n<block->dims;n++) dat->d_p[n] = d_p[n];

  for(int n=block->dims; n < OPS_MAX_DIM;n++) {
    dat->size[n] = 1;
    dat->base[n] = 0;
    dat->d_m[n] = 0;
    dat->d_p[n] = 0;
  }

  dat->data = (char *)data;
  dat->data_d = NULL;
  dat->user_managed = 1;
  dat->dirty_hd = 0;
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
  //Double_linked_list test; //Head of the double linked list
  //add item to the end of the list
  TAILQ_INSERT_TAIL(&OPS_dat_list, item, entries);
  OPS_dat_index++;

  //Another entry for a different list
  item = (ops_dat_entry *)malloc(sizeof(ops_dat_entry));
  if (item == NULL) {
    printf ( " op_decl_dat error -- error allocating memory to double linked list entry\n" );
    exit ( -1 );
  }
  item->dat = dat;
  TAILQ_INSERT_TAIL(&OPS_block_list[block->index].datasets, item, entries);
  OPS_block_list[block->index].num_datasets++;

  return dat;
}


ops_dat ops_decl_dat_temp_core ( ops_block block, int dim,
  int *dataset_size, int* base, int* d_m, int* d_p, char * data, int type_size, char const * type, char const * name )
{
  //Check if this dat already exists in the double linked list
  ops_dat found_dat = search_dat(block, dim, dataset_size, base, type, name);
  if ( found_dat != NULL) {
    printf("ops_dat with name %s already exists, cannot create temporary ops_dat\n ", name);
    exit(2);
  }
  //if not found ...
  return ops_decl_dat_core ( block, dim, dataset_size, base, d_m, d_p, data, type_size, type, name );
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
  stencil->name = copy_str(name);

  stencil->stencil = (int *)xmalloc(dims*points*sizeof(int));
  memcpy(stencil->stencil,sten,sizeof(int)*dims*points);

  //for (int i = 0; i < dims*points; i++)
  //  printf("%d ",stencil->stencil[i]);
  //printf("\n");

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

ops_arg ops_arg_reduce_core ( ops_reduction handle, int dim, const char *type, ops_access acc) {
  ops_arg arg;
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = (char*)handle;
  arg.acc = acc;
  if (handle->initialized == 0) {
    handle->initialized = 1;
    handle->acc = acc;
    if (acc == OPS_INC) memset(handle->data, 0, handle->size);
    if (strcmp(type,"double")==0) { //TODO: handle other types
      if (acc == OPS_MIN) for (int i = 0; i < handle->size/8; i++) ((double*)handle->data)[i] = DBL_MAX;
      if (acc == OPS_MAX) for (int i = 0; i < handle->size/8; i++) ((double*)handle->data)[i] = -1.0*DBL_MAX;
    }
    else if (strcmp(type,"float")==0) {
      if (acc == OPS_MIN) for (int i = 0; i < handle->size/4; i++) ((double*)handle->data)[i] = FLT_MAX;
      if (acc == OPS_MAX) for (int i = 0; i < handle->size/4; i++) ((double*)handle->data)[i] = -1.0f*FLT_MAX;
    }
    else if (strcmp(type,"int")==0) {
      if (acc == OPS_MIN) for (int i = 0; i < handle->size/4; i++) ((double*)handle->data)[i] = INT_MAX;
      if (acc == OPS_MAX) for (int i = 0; i < handle->size/4; i++) ((double*)handle->data)[i] = -1*INT_MAX;
    }
  } else if (handle->acc != acc) {
    printf("ops_reduction handle %s was aleady used with a different access type\n",handle->name);
    exit(-1);
  }
  return arg;
}

ops_halo_group ops_decl_halo_group(int nhalos, ops_halo *halos) {
  if ( OPS_halo_group_index == OPS_halo_group_max ) {
    OPS_halo_group_max += 10;
    OPS_halo_group_list = (ops_halo_group *) realloc(OPS_halo_group_list,OPS_halo_group_max * sizeof(ops_halo_group));

    if ( OPS_halo_group_list == NULL ) {
      printf ( " ops_decl_halo_group error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }
  ops_halo_group grp = (ops_halo_group)xmalloc(sizeof(ops_halo_group_core));
  grp->nhalos = nhalos;
  grp->halos = halos; //TODO: make a copy?
  grp->index = OPS_halo_group_index;
  OPS_halo_group_list[OPS_halo_group_index++] = grp;

  return grp;
}

ops_halo ops_decl_halo_core(ops_dat from, ops_dat to, int *iter_size, int* from_base, int *to_base, int *from_dir, int *to_dir) {
  if ( OPS_halo_index == OPS_halo_max ) {
    OPS_halo_max += 10;
    OPS_halo_list = (ops_halo *) realloc(OPS_halo_list,OPS_halo_max * sizeof(ops_halo));

    if ( OPS_halo_list == NULL ) {
      printf ( " ops_decl_halo_core error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_halo halo = (ops_halo)xmalloc(sizeof(ops_halo_core));
  halo->index = OPS_halo_index;
  halo->from = from;
  halo->to = to;
  for (int i = 0; i < from->block->dims; i++) {
    halo->iter_size[i] = iter_size[i];
    halo->from_base[i] = from_base[i];
    halo->to_base[i] = to_base[i];
    halo->from_dir[i] = from_dir[i];
    halo->to_dir[i] = to_dir[i];
  }
  for (int i = from->block->dims; i < OPS_MAX_DIM; i++) {
    halo->iter_size[i] = 1;
    halo->from_base[i] = 0;
    halo->to_base[i] = 0;
    halo->from_dir[i] = i+1;
    halo->to_dir[i] = i+1;
  }

  OPS_halo_list[OPS_halo_index++] = halo;
  return halo;
}

ops_arg ops_arg_dat_core ( ops_dat dat, ops_stencil stencil, ops_access acc ) {
  ops_arg arg;
  arg.argtype = OPS_ARG_DAT;
  arg.dat = dat;
  arg.stencil = stencil;
  if ( dat != NULL ) {
    arg.data = dat->data;
    arg.data_d = dat->data_d;
  } else {
    arg.data = NULL;
    arg.data_d = NULL;
  }
  arg.acc = acc;
  arg.opt = 1;
  return arg;
}

ops_arg ops_arg_gbl_core ( char * data, int dim, int size, ops_access acc ) {
  ops_arg arg;
  arg.argtype = OPS_ARG_GBL;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = dim;
  arg.data = data;
  arg.acc = acc;
  return arg;
}

ops_arg ops_arg_idx () {
  ops_arg arg;
  arg.argtype = OPS_ARG_IDX;
  arg.dat = NULL;
  arg.data_d = NULL;
  arg.stencil = NULL;
  arg.dim = 0;
  arg.data = NULL;
  arg.acc = 0;
  return arg;
}

ops_reduction ops_decl_reduction_handle_core(int size, const char *type, const char *name) {
  if ( OPS_reduction_index == OPS_reduction_max ) {
    OPS_reduction_max += 10;
    OPS_reduction_list = (ops_reduction *) realloc(OPS_reduction_list,OPS_reduction_max * sizeof(ops_reduction));

    if ( OPS_reduction_list == NULL ) {
      printf ( " ops_decl_reduction_handle error -- error reallocating memory\n" );
      exit ( -1 );
    }
  }

  ops_reduction red = (ops_reduction)malloc(sizeof(ops_reduction_core));
  red->initialized = 0;
  red->size = size;
  red->data = (char *)malloc(size*sizeof(char));
  red->name = copy_str(name);
  red->type = copy_str(type);
  OPS_reduction_list[OPS_reduction_index] = red;
  red->index = OPS_reduction_index++;
  return red;
}

void ops_diagnostic_output ( )
{
  if ( OPS_diags > 1 ) {
    printf ( "\n OPS diagnostic output\n" );
    printf ( " --------------------\n" );

    printf ( "\n block dimension\n" );
    printf ( " -------------------\n" );
    for ( int n = 0; n < OPS_block_index; n++ ) {
      printf ( " %15s %15dD ", OPS_block_list[n].block->name, OPS_block_list[n].block->dims );
      printf("\n");
    }

    printf ( "\n dats item/point [block_size] [base][d_m][d_p]  block\n" );
    printf ( " ------------------------------\n" );
    ops_dat_entry *item;
    TAILQ_FOREACH(item, &OPS_dat_list, entries) {
      printf ( " %15s %15d ", (item->dat)->name, (item->dat)->dim );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->size[i] );
      printf ( " " );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->base[i] );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->d_m[i] );
      for (int i=0; i<(item->dat)->block->dims; i++)
        printf ( "[%d]",(item->dat)->d_p[i] );

      printf ( " %15s\n", (item->dat)->block->name );
    }
    printf ( "\n" );
  }
}


void ops_dump3(ops_dat dat, const char* name) {
  //TODO: this has to be backend-specific
/*  char str[100];
  strcpy(str,"./dump/");
  strcat(str,name);
  strcat(str,"_");
  strcat(str,dat->name);
  //const char* file_name = dat->name;
  FILE *fp;
  if ( (fp = fopen(str,"w")) == NULL) {
    printf("can't open file %s\n",str);
    exit(2);
  }
  int x_end = dat->tail[0]==-3 ? dat->block_size[0]+dat->tail[0] : dat->block_size[0]+dat->tail[0]-1;
  int y_end = dat->tail[1]==-3 ? dat->block_size[1]+dat->tail[1] : dat->block_size[1]+dat->tail[1]-1;
  int z_end = dat->tail[2]==-3 ? dat->block_size[2]+dat->tail[2] : dat->block_size[2]+dat->tail[2]-1;
  for (int z = -dat->offset[2]; z < z_end; z++) {
    for (int y = -dat->offset[1]; y < y_end; y++) {
      for (int x = -dat->offset[0]; x < x_end; x++) {
        fprintf(fp,"%d %d %d %.17g\n",x+dat->offset[0],y+dat->offset[1],z+dat->offset[2],
          *(double*)(dat->data+8*(x+dat->block_size[0]*y+dat->block_size[1]*dat->block_size[0]*z)));

      }
    }
  }
  fclose(fp);
  */
}

void ops_print_dat_to_txtfile_core(ops_dat dat, const char* file_name)
{
  //TODO: this has to be backend-specific
  FILE *fp;
  if ( (fp = fopen(file_name,"a")) == NULL) {
    printf("can't open file %s\n",file_name);
    exit(2);
  }

  if (fprintf(fp,"ops_dat:  %s \n", dat->name)<0) {
    printf("error writing to %s\n",file_name);
    exit(2);
  }
  if (fprintf(fp,"ops_dat dim:  %d \n", dat->dim)<0) {
    printf("error writing to %s\n",file_name);
    exit(2);
  }

  if (fprintf(fp,"block Dims : %d ", dat->block->dims)<0) {
      printf("error writing to %s\n",file_name);
      exit(2);
    }

  for(int i = 0; i < dat->block->dims; i++) {
    if (fprintf(fp,"[%d]", dat->size[i])<0) {
      printf("error writing to %s\n",file_name);
      exit(2);
    }
  }
  fprintf(fp,"\n");
  

  if (fprintf(fp,"elem size %d \n", dat->elem_size)<0) {
    printf("error writing to %s\n",file_name);
    exit(2);
  }

  if(dat->block->dims == 3) {
    if( strcmp(dat->type,"double") == 0 ) {
      for(int i = 0; i < dat->size[2]; i++ ) {
        for(int j = 0; j < dat->size[1]; j++ ) {
          for(int k = 0; k < dat->size[0]; k++ ) {
            if (fprintf(fp, " %3.10lf",
              ((double *)dat->data)[i*dat->size[1]*dat->size[0]+
                                    j*dat->size[0]+k])<0) {
              printf("error writing to %s\n",file_name);
              exit(2);
              }
          }
          fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
      }
    }
    else if( strcmp(dat->type,"float") == 0 ) {
      for(int i = 0; i < dat->size[2]; i++ ) {
        for(int j = 0; j < dat->size[1]; j++ ) {
          for(int k = 0; k < dat->size[0]; k++ ) {
            if (fprintf(fp, "%e ", ((float *)dat->data)[i*dat->size[1]*dat->size[0]+
                                      j*dat->size[0]+k])<0) {
              printf("error writing to %s\n",file_name);
              exit(2);
              }
          }
          fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
      }
    }
    else if( strcmp(dat->type,"int") == 0 ) {
      for(int i = 0; i < dat->size[2]; i++ ) {
        for(int j = 0; j < dat->size[1]; j++ ) {
          for(int k = 0; k < dat->size[0]; k++ ) {
            if (fprintf(fp, "%d ", ((int *)dat->data)[i*dat->size[1]*dat->size[0]+
                                      j*dat->size[0]+k])<0) {
              printf("error writing to %s\n",file_name);
              exit(2);
            }
          }
          fprintf(fp,"\n");
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
  else if(dat->block->dims == 2) {
    if( strcmp(dat->type,"double") == 0 ) {
      for(int i = 0; i < dat->size[1]; i++ ) {
        for(int j = 0; j < dat->size[0]; j++ ) {
          for(int d = 0; d < 2; d++ ) {
            printf("%d,%d,%d ",i,j,d);
            //if (fprintf(fp, " %3.10lf",((double *)dat->data)[i*dat->size[0]*dat->dim+j*dat->dim+d])<0) {
            if (fprintf(fp, " %3.10lf",((double *)dat->data)[(i*dat->size[0]+j)*2+d])<0) {
              printf("error writing to %s\n",file_name);
              exit(2);
            }
          }
        }
        fprintf(fp,"\n");
        printf("\n");
      }
    }
    else if( strcmp(dat->type,"float") == 0 ) {
      for(int i = 0; i < dat->size[1]; i++ ) {
        for(int j = 0; j < dat->size[0]; j++ ) {
          if (fprintf(fp, "%e ", ((float *)dat->data)[i*dat->size[0]+j])<0) {
            printf("error writing to %s\n",file_name);
            exit(2);
          }
        }
        fprintf(fp,"\n");
      }
    }
    else if( strcmp(dat->type,"int") == 0 ) {
      for(int i = 0; i < dat->size[1]; i++ ) {
        for(int j = 0; j < dat->size[0]; j++ ) {
          if (fprintf(fp, "%d ", ((int *)dat->data)[i*dat->size[0]+j])<0) {
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
      for(int j = 0; j < dat->size[0]; j++ ) {
        if (fprintf(fp, "%3.10lf ", ((double *)dat->data)[j])<0) {
          printf("error writing to %s\n",file_name);
          exit(2);
        }
      }
      fprintf(fp,"\n");
    }
    else if( strcmp(dat->type,"float") == 0 ) {
      for(int j = 0; j < dat->size[0]; j++ ) {
        if (fprintf(fp, "%e ", ((float *)dat->data)[j])<0) {
          printf("error writing to %s\n",file_name);
          exit(2);
        }
      }
      fprintf(fp,"\n");
    }
    else if( strcmp(dat->type,"int") == 0 ) {
      for(int j = 0; j < dat->size[0]; j++ ) {
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

void ops_timing_output(FILE *stream)
{
  if ( OPS_diags > 1 ) {
    int maxlen = 0;
    for (int i = 0; i < OPS_kern_max; i++) {
      if (OPS_kernels[i].count > 0) maxlen = MAX(maxlen, strlen(OPS_kernels[i].name));
      if (OPS_kernels[i].count > 0 && strlen(OPS_kernels[i].name)>50) {
        printf("Too long\n");
      }
    }
    char *buf = (char*)malloc((maxlen+180)*sizeof(char));
    char buf2[180];
    sprintf(buf,"Name");
    for (int i = 4; i < maxlen;i++) strcat(buf," ");
    ops_fprintf(stream,"\n\n%s  Count Time     MPI-time Bandwidth (GB/s)\n",buf);

    sprintf(buf,"");
    for (int i = 0; i < maxlen+31;i++) strcat(buf,"-");
    ops_fprintf(stream,"%s\n",buf);
    double sumtime = 0.0f;
    for (int k = 0; k < OPS_kern_max; k++) {
      if (OPS_kernels[k].count < 1) continue;
      sprintf(buf,"%s",OPS_kernels[k].name);
      for (int i = strlen(OPS_kernels[k].name); i < maxlen+2; i++) strcat(buf," ");

      double moments_mpi_time[2] = {0.0};
      double moments_time[2] = {0.0};
      ops_compute_moment(OPS_kernels[k].time, &moments_time[0], &moments_time[1]);
      ops_compute_moment(OPS_kernels[k].mpi_time, &moments_mpi_time[0], &moments_mpi_time[1]);

      sprintf(buf2,"%-5d %-6f (%-6f) %-6f (%-6f)  %-13.2f", OPS_kernels[k].count, moments_time[0],
        sqrt(moments_time[1] - moments_time[0]*moments_time[0]),
        moments_mpi_time[0], sqrt(moments_mpi_time[1] - moments_mpi_time[0]*moments_mpi_time[0]),
        OPS_kernels[k].transfer/OPS_kernels[k].time/1000/1000/1000);

      //sprintf(buf2,"%-5d %-6f  %-6f  %-13.2f", OPS_kernels[k].count, OPS_kernels[k].time,
      //  OPS_kernels[k].mpi_time, OPS_kernels[k].transfer/OPS_kernels[k].time/1000/1000/1000);
      ops_fprintf(stream,"%s%s\n",buf,buf2);
      sumtime += OPS_kernels[k].time;
    }
    ops_fprintf(stream,"Total kernel time: %g\n",sumtime);
    //printf("Times: %g %g %g\n",ops_gather_time, ops_sendrecv_time, ops_scatter_time);
    free(buf);
  }
}

void ops_timers_core( double * cpu, double * et )
{
  (void)cpu;
  struct timeval t;

  gettimeofday ( &t, ( struct timezone * ) 0 );
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}

void
ops_timing_realloc ( int kernel, const char *name )
{
  int OPS_kern_max_new;
  OPS_kern_curr = kernel;

  if ( kernel >= OPS_kern_max )
  {
    OPS_kern_max_new = kernel + 10;
    OPS_kernels = ( ops_kernel * ) realloc ( OPS_kernels, OPS_kern_max_new * sizeof ( ops_kernel ) );
    if ( OPS_kernels == NULL )
    {
      printf ( " ops_timing_realloc error \n" );
      exit ( -1 );
    }

    for ( int n = OPS_kern_max; n < OPS_kern_max_new; n++ )
    {
      OPS_kernels[n].count = -1;
      OPS_kernels[n].time = 0.0f;
      OPS_kernels[n].transfer = 0.0f;
      OPS_kernels[n].mpi_time = 0.0f;
    }
    OPS_kern_max = OPS_kern_max_new;
  }

  if (OPS_kernels[kernel].count == -1) {
    OPS_kernels[kernel].name = (char *)malloc((strlen(name)+1)*sizeof(char));
    strcpy(OPS_kernels[kernel].name,name);
    OPS_kernels[kernel].count = 0;
  }
}

float ops_compute_transfer(int dims, int *range, ops_arg *arg) {
  float size = 1.0f;
  for (int i = 0; i < dims; i++) {
    if (arg->stencil->stride[i] != 0)
      size *= (range[2*i+1]-range[2*i]);
  }
  size *= arg->dat->elem_size*((arg->argtype==OPS_READ || arg->argtype==OPS_WRITE) ? 1.0f : 2.0f);
  return size;
}

void ops_register_args(ops_arg *args, const char *name) {
  OPS_curr_args = args;
  OPS_curr_name = name;
}

int ops_stencil_check_2d(int arg_idx, int idx0, int idx1, int dim0, int dim1) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[2*i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[2*i+1] == idx1) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d) not found in declaration %s in loop %s arg %d\n",
             idx0, idx1, OPS_curr_args[arg_idx].stencil->name, OPS_curr_name, arg_idx);
      exit(-1);
    }
  }
  return idx0+dim0*(idx1);
}
int ops_stencil_check_3d(int arg_idx, int idx0, int idx1, int idx2, int dim0, int dim1) {
  if (OPS_curr_args) {
    int match = 0;
    for (int i = 0; i < OPS_curr_args[arg_idx].stencil->points; i++) {
      if (OPS_curr_args[arg_idx].stencil->stencil[3*i] == idx0 &&
          OPS_curr_args[arg_idx].stencil->stencil[3*i+1] == idx1 &&
          OPS_curr_args[arg_idx].stencil->stencil[3*i+2] == idx2) {
        match = 1;
        break;
      }
    }
    if (match == 0) {
      printf("Error: stencil point (%d,%d,%d) not found in declaration %s in loop %s arg %d\n",
             idx0, idx1, idx2, OPS_curr_args[arg_idx].stencil->name, OPS_curr_name, arg_idx);
      exit(-1);
    }
  }
  return idx0+dim0*(idx1)+dim0*dim1*(idx2);
}
