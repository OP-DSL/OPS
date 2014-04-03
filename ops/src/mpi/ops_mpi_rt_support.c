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

/** @brief ops mpi run-time support routines
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implements the runtime support routines for the OPS mpi backend
  */

#include <ops_lib_core.h>

#include <mpi.h>
#include <ops_mpi_core.h>

extern int OPS_diags;

// Timing
double t1,t2,c1,c2;

int ops_is_root()
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  return (my_rank==MPI_ROOT);
}

/*fixed depth halo exchange*/
void ops_exchange_halo(ops_arg* arg, int d/*depth*/)
{
  ops_dat dat = arg->dat;
  int any_dirty = dat->dirtybit;
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  for(int n=0;n<sb->ndim;n++)
    for (int d1 = 0; d1 <= d; d1++)
      any_dirty = any_dirty || 
                  dat->dirty_dir_send[2*MAX_DEPTH*n + MAX_DEPTH + d1] || dat->dirty_dir_send[2*MAX_DEPTH*n + d1] ||
                  dat->dirty_dir_recv[2*MAX_DEPTH*n + MAX_DEPTH + d1] || dat->dirty_dir_recv[2*MAX_DEPTH*n + d1];

  if(arg->opt == 1 && any_dirty) { //need to check OPS accs

    // ops_printf("exchanging %s\n",arg->dat->name);

    //sub_block_list sb = OPS_sub_block_list[dat->block->index];
    sub_dat_list sd = OPS_sub_dat_list[dat->index];

    int i1,i2,i3,i4; //indicies for halo and boundary of the dat
    int* d_m = sd->d_m;
    int* d_p = sd->d_p;
    int* prod = sd->prod;
    int size = dat->size;
    MPI_Status status;

    for(int n=0;n<sb->ndim;n++){
      if(dat->block_size[n] > 1 && d > 0) {

        i1 = (-d_m[n] - d) * prod[n-1];
        i2 = (-d_m[n]    ) * prod[n-1];
        i3 = (prod[n]/prod[n-1] - (-d_p[n]) - d) * prod[n-1];
        i4 = (prod[n]/prod[n-1] - (-d_p[n])    ) * prod[n-1];

        //send in positive direction, receive from negative direction
        //printf("Exchaning 1 From:%d To: %d\n", i3, i1);
        MPI_Sendrecv(&dat->data[i3*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_p[n],0,
                     &dat->data[i1*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_m[n],0,
                     OPS_CART_COMM, &status);
        for (int d1 = 0; d1 <= d; d1++)  dat->dirty_dir_send[2*MAX_DEPTH*n + MAX_DEPTH + d1] = 0;
        for (int d1 = 0; d1 <= d; d1++)  dat->dirty_dir_recv[2*MAX_DEPTH*n + d1] = 0;

        //send in negative direction, receive from positive direction
        //printf("Exchaning 2 From:%d To: %d\n", i2, i4);
        MPI_Sendrecv(&dat->data[i2*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_m[n],1,
                     &dat->data[i4*size],1,sd->mpidat[MAX_DEPTH*n+d],sb->id_p[n],1,
                     OPS_CART_COMM, &status);
        for (int d1 = 0; d1 <= d; d1++)  dat->dirty_dir_send[2*MAX_DEPTH*n + d1] = 0;
        for (int d1 = 0; d1 <= d; d1++)  dat->dirty_dir_recv[2*MAX_DEPTH*n + MAX_DEPTH + d1] = 0;
      }
    }

    dat->dirtybit = 0;
  }

}

/*dynamic depth halo exchange*/
void ops_exchange_halo2(ops_arg* arg, int* d_pos, int* d_neg /*depth*/)
{
  ops_dat dat = arg->dat;

  if(arg->opt == 1) { //need to check OPS accs
  //if( dat->dirtybit == 1) { //need to check OPS accs
  // ops_printf("exchanging %s\n",arg->dat->name);

    sub_block_list sb = OPS_sub_block_list[dat->block->index];
    sub_dat_list sd = OPS_sub_dat_list[dat->index];

    int i1,i2,i3,i4; //indicies for halo and boundary of the dat
    int* d_m = sd->d_m;
    int* d_p = sd->d_p;
    int* prod = sd->prod;
    int size = dat->size;
    MPI_Status status;

    for(int n=0;n<sb->ndim;n++){

      int d_min = abs(d_neg[n]);
      //if (d_min > 2 || d_pos[n] > 2) printf("big stencil in %s\n",dat->name);
      //d_pos[n] = 2; //hard coded for now .. change for dynamic halo depth
      if(dat->block_size[n] > 1 ) {//&& (d_pos[n] > 0 || d_min > 0) ) {

        int actual_depth = 0;
        for (int d = 0; d <= d_min; d++)
          if(dat->dirty_dir_send[2*MAX_DEPTH*n + MAX_DEPTH + d] == 1 || dat->dirty_dir_recv[2*MAX_DEPTH*n + d] == 1) actual_depth = d;

        i1 = (-d_m[n] - actual_depth) * prod[n-1];
        i3 = (prod[n]/prod[n-1] - (-d_p[n]) - actual_depth) * prod[n-1];
        //printf("Exchange %s, dim %d min depth %d req %d\n",dat->name, n, actual_depth, d_min);

        if (OPS_diags>4) { //Consistency checking
          int they_send;
          MPI_Sendrecv(&actual_depth,1,MPI_INT,sb->id_p[n],666,
            &they_send,1,MPI_INT,sb->id_m[n],666,OPS_CART_COMM, &status);
          if (sb->id_m[n]>=0 && actual_depth != they_send) {
            printf("Left recv mismatch\n");
            MPI_Abort(OPS_CART_COMM,-1);
          }
        }
        
        //send in positive direction, receive from negative direction
        //printf("Exchaning 1 From:%d To: %d\n", i3, i1);
        if(actual_depth > 0)
        MPI_Sendrecv(&dat->data[i3*size],1,sd->mpidat[MAX_DEPTH*n+actual_depth],sb->id_p[n],sb->ndim+n,
                     &dat->data[i1*size],1,sd->mpidat[MAX_DEPTH*n+actual_depth],sb->id_m[n],sb->ndim+n,
                     OPS_CART_COMM, &status);
        for (int d = 0; d <= actual_depth; d++) dat->dirty_dir_send[2*MAX_DEPTH*n + MAX_DEPTH + d] = 0;
        for (int d = 0; d <= actual_depth; d++) dat->dirty_dir_recv[2*MAX_DEPTH*n + d] = 0;

        actual_depth = 0;
        for (int d = 0; d <= d_pos[n]; d++)
          if(dat->dirty_dir_send[2*MAX_DEPTH*n + d] == 1 || dat->dirty_dir_recv[2*MAX_DEPTH*n + MAX_DEPTH + d] == 1) actual_depth = d;

        i2 = (-d_m[n]    ) * prod[n-1];
        i4 = (prod[n]/prod[n-1] - (-d_p[n])    ) * prod[n-1];
        //printf("Exchange %s, dim %d max depth %d req %d\n",dat->name, n, actual_depth, d_pos[n]);

        if (OPS_diags>4) { //Consistency checking
          int they_send;
          MPI_Sendrecv(&actual_depth,1,MPI_INT,sb->id_m[n],665,
            &they_send,1,MPI_INT,sb->id_p[n],665,OPS_CART_COMM, &status);
          if (sb->id_p[n]>=0 && actual_depth != they_send) {
            printf("Right recv mismatch\n");
            MPI_Abort(OPS_CART_COMM,-1);
          }
        }
        
        //send in negative direction, receive from positive direction
        //printf("Exchaning 2 From:%d To: %d\n", i2, i4);
        if(actual_depth > 0)
        MPI_Sendrecv(&dat->data[i2*size],1,sd->mpidat[MAX_DEPTH*n+actual_depth],sb->id_m[n],n,
                     &dat->data[i4*size],1,sd->mpidat[MAX_DEPTH*n+actual_depth],sb->id_p[n],n,
                     OPS_CART_COMM, &status);

        for (int d = 0; d <= actual_depth; d++) dat->dirty_dir_send[2*MAX_DEPTH*n + d] = 0;
        for (int d = 0; d <= actual_depth; d++) dat->dirty_dir_recv[2*MAX_DEPTH*n + MAX_DEPTH + d] = 0;
      }
    }
  }
}

int intersection(int range1_beg, int range1_end, int range2_beg, int range2_end) {
  int i_min = MAX(range1_beg,range2_beg);
  int i_max = MIN(range1_end,range2_end);
  return i_max>i_min ? i_max-i_min : 0;
}

int contains(int point, int* range) {
  return (point >= range[0] && point < range[1]);
}

void ops_exchange_halo3(ops_arg* arg, int* d_pos, int* d_neg /*depth*/, int *iter_range) {
  ops_dat dat = arg->dat;
//????
//TODO: diagonal dependency????
//????
  if(arg->opt == 1) { //need to check OPS accs

    sub_block_list sb = OPS_sub_block_list[dat->block->index];
    sub_dat_list sd = OPS_sub_dat_list[dat->index];
    int left_send_depth[OPS_MAX_DIM] = {0};
    int left_recv_depth[OPS_MAX_DIM] = {0};
    int right_send_depth[OPS_MAX_DIM] = {0};
    int right_recv_depth[OPS_MAX_DIM] = {0};

    int range_intersect[OPS_MAX_DIM] = {0};

    int* d_m = sd->d_m;
    int* d_p = sd->d_p;
    int* prod = sd->prod;
    int size = dat->size;
    MPI_Status status;
    int ndim = sb->ndim;

    for (int dim = 0; dim < ndim; dim++) {
      range_intersect[dim] = intersection( iter_range[2*dim]+d_neg[dim],
                                           iter_range[2*dim+1]+d_pos[dim],
                                           sb->istart[dim],
                                           sb->iend[dim]+1); //i.e. the intersection of the dependency range with my full range
                                            
      if (d_pos[dim]>0) {
        left_send_depth [dim] = contains(sb->istart[dim]-1,&iter_range[2*dim]) ?          //if my left neighbor's last point is in the iteration range
                                d_pos[dim] :                                              //then it needs full depth required by the stencil
                                (iter_range[2*dim+1]<sb->istart[dim] ?                    //otherwise if range ends somewhere before my range begins
                                MAX(0,d_pos[dim]-(sb->istart[dim]-iter_range[2*dim+1])) : //the dependency may still reach into my range
                                0);                                                       //otherwise 0
        
        right_recv_depth [dim]= contains(sb->iend[dim]+1-1,&iter_range[2*dim]) ?          //if my last point is in the iteration range
                                d_pos[dim] :                                              //then I need full depth required by the stencil
                                (iter_range[2*dim+1]<sb->iend[dim]+1 ?                    //otherwise if range ends somewhere before my range ends
                                MAX(0,d_pos[dim]-(sb->iend[dim]+1-iter_range[2*dim+1])) : //the dependency may still reach into my neighbor's range
                                0);                                                       //otherwise 0
      }

      if (d_neg[dim]<0) {
        left_recv_depth [dim] = contains(sb->istart[dim],&iter_range[2*dim]) ?            //if my first point is in the iteration range
                                -d_neg[dim] :                                             //then I need full depth required by the stencil
                                (iter_range[2*dim]>sb->istart[dim] ?                      //otherwise if range starts somewhere after my range begins
                                MAX(0,-d_neg[dim]-(iter_range[2*dim]-sb->istart[dim])) :  //the dependency may still reach into my left neighbor's range
                                0);                                                       //otherwise 0

        right_send_depth [dim]= contains(sb->iend[dim]+1,&iter_range[2*dim]) ?            //if my neighbor's first point is in the iteration range
                                -d_neg[dim] :                                             //then it needs full depth required by the stencil
                                (iter_range[2*dim]>sb->iend[dim]+1 ?                      //otherwise if range starts somewhere after my neighbor's range begins
                                MAX(0,-d_neg[dim]-(iter_range[2*dim]-sb->iend[dim]-1)) :  //the dependency may still reach into my range
                                0);                                                       //otherwise 0
      }
    }

    for (int dim =0;dim<ndim;dim++) {
      if(dat->block_size[dim] <= 1 ) continue;

      //decide whether we intersect in all other dimensions
      int other_dims = 1;
      for (int d2 = 0; d2 < ndim; d2++)
        if (d2 != dim) other_dims = other_dims && range_intersect[d2]>0;

      if (other_dims == 0) continue;

      //negative direction
      
      //decide actual depth based on dirtybits
      int actual_depth_send = 0;
      for (int d = 0; d <= left_send_depth[dim]; d++)
        if(dat->dirty_dir_send[2*MAX_DEPTH*dim + d] == 1) actual_depth_send = d;
      
      int actual_depth_recv = 0;
      for (int d = 0; d <= right_recv_depth[dim]; d++)
        if(dat->dirty_dir_recv[2*MAX_DEPTH*dim + MAX_DEPTH + d] == 1) actual_depth_recv = d;
      
      //set up initial pointers
      int i2 = (-d_m[dim]    ) * prod[dim-1];
      int i4 = (prod[dim]/prod[dim-1] - (-d_p[dim])    ) * prod[dim-1];

      if (OPS_diags>4) { //Consistency checking
        int they_send;
        MPI_Sendrecv(&actual_depth_send,1,MPI_INT,sb->id_m[dim],665,
          &they_send,1,MPI_INT,sb->id_p[dim],665,OPS_CART_COMM, &status);
        if (sb->id_p[dim]>=0 && actual_depth_recv != they_send) {
          printf("Right recv mismatch\n");
          MPI_Abort(OPS_CART_COMM,-1);
        }
      }
      //negative direction exchange
      MPI_Sendrecv(&dat->data[i2*size],1,sd->mpidat[MAX_DEPTH*dim+actual_depth_send],sb->id_m[dim],dim,
        &dat->data[i4*size],1,sd->mpidat[MAX_DEPTH*dim+actual_depth_recv],sb->id_p[dim],dim,
        OPS_CART_COMM, &status);
        
      //clear dirtybits
      for (int d = 0; d <= actual_depth_send; d++) dat->dirty_dir_send[2*MAX_DEPTH*dim + d] = 0;
      for (int d = 0; d <= actual_depth_recv; d++) dat->dirty_dir_recv[2*MAX_DEPTH*dim + MAX_DEPTH + d] = 0;


      //similarly for positive direction
      
      //decide actual depth based on dirtybits
      actual_depth_send = 0;
      for (int d = 0; d <= right_send_depth[dim]; d++)
        if(dat->dirty_dir_send[2*MAX_DEPTH*dim + MAX_DEPTH + d] == 1) actual_depth_send = d;
      
      actual_depth_recv = 0;
      for (int d = 0; d <= left_recv_depth[dim]; d++)
        if(dat->dirty_dir_recv[2*MAX_DEPTH*dim + d] == 1) actual_depth_recv = d;

      //set up initial pointers
      int i1 = (-d_m[dim] - actual_depth_recv) * prod[dim-1];
      int i3 = (prod[dim]/prod[dim-1] - (-d_p[dim]) - actual_depth_send) * prod[dim-1];

      if (OPS_diags>4) { //Consistency checking
        int they_send;
        MPI_Sendrecv(&actual_depth_send,1,MPI_INT,sb->id_p[dim],666,
          &they_send,1,MPI_INT,sb->id_m[dim],666,OPS_CART_COMM, &status);
        if (sb->id_m[dim]>=0 && actual_depth_recv != they_send) {
          printf("Left recv mismatch\n");
          MPI_Abort(OPS_CART_COMM,-1);
        }
      }
      
      //positive direction exchange
      MPI_Sendrecv(&dat->data[i3*size],1,sd->mpidat[MAX_DEPTH*dim+actual_depth_send],sb->id_p[dim],ndim+dim,
        &dat->data[i1*size],1,sd->mpidat[MAX_DEPTH*dim+actual_depth_recv],sb->id_m[dim],ndim+dim,
        OPS_CART_COMM, &status);

      //clear dirtybits
      for (int d = 0; d <= actual_depth_send; d++) dat->dirty_dir_send[2*MAX_DEPTH*dim + MAX_DEPTH + d] = 0;
      for (int d = 0; d <= actual_depth_recv; d++) dat->dirty_dir_recv[2*MAX_DEPTH*dim + d] = 0;
    }
  }
}

void ops_mpi_reduce_double(ops_arg* arg, double* data)
{
  (void)data;
  //if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
    double result[arg->dim*ops_comm_size];

    if(arg->acc == OPS_INC)//global reduction
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_SUM, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MAX)//global maximum
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MAX, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MIN)//global minimum
      MPI_Allreduce((double *)arg->data, result, arg->dim, MPI_DOUBLE, MPI_MIN, OPS_MPI_WORLD);
    else if(arg->acc == OPS_WRITE) {//any
      MPI_Allgather((double *)arg->data, arg->dim, MPI_DOUBLE, result, arg->dim, MPI_DOUBLE, OPS_MPI_WORLD);
      for (int i = 1; i < ops_comm_size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0)
            result[j] = result[i*arg->dim+j];
        }
      }
    }
    memcpy(arg->data, result, sizeof(double)*arg->dim);
  //}
}



void ops_mpi_reduce_float(ops_arg* arg, float* data)
{
  (void)data;

  //if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
    float result[arg->dim*ops_comm_size];

    if(arg->acc == OPS_INC)//global reduction
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_SUM, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MAX)//global maximum
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MAX, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MIN)//global minimum
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MIN, OPS_MPI_WORLD);
    else if(arg->acc == OPS_WRITE) {//any
      MPI_Allgather((float *)arg->data, arg->dim, MPI_FLOAT, result, arg->dim, MPI_FLOAT, OPS_MPI_WORLD);
      for (int i = 1; i < ops_comm_size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0.0f)
            result[j] = result[i*arg->dim+j];
        }
      }
    }
    memcpy(arg->data, result, sizeof(float)*arg->dim);
  //}
}


void ops_mpi_reduce_int(ops_arg* arg, int* data)
{
  (void)data;

  //if(arg->argtype == OPS_ARG_GBL && arg->acc != OPS_READ) {
    int result[arg->dim*ops_comm_size];

    if(arg->acc == OPS_INC)//global reduction
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_SUM, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MAX)//global maximum
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_MAX, OPS_MPI_WORLD);
    else if(arg->acc == OPS_MIN)//global minimum
      MPI_Allreduce((int *)arg->data, result, arg->dim, MPI_INT, MPI_MIN, OPS_MPI_WORLD);
    else if(arg->acc == OPS_WRITE) {//any
      MPI_Allgather((int *)arg->data, arg->dim, MPI_INT, result, arg->dim, MPI_INT, OPS_MPI_WORLD);
      for (int i = 1; i < ops_comm_size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i*arg->dim+j] != 0)
            result[j] = result[i*arg->dim+j];
        }
      }
    }
    memcpy(arg->data, result, sizeof(int)*arg->dim);
  //}
}

void ops_set_halo_dirtybit(ops_arg *arg)
{
  if(arg->opt == 0) return;
  arg->dat->dirtybit = 1;
  for(int i = 0; i<2*arg->dat->block->dims*MAX_DEPTH;i++) arg->dat->dirty_dir_send[i] = 1;
  for(int i = 0; i<2*arg->dat->block->dims*MAX_DEPTH;i++) arg->dat->dirty_dir_recv[i] = 1;
}

void ops_set_halo_dirtybit3(ops_arg *arg, int *iter_range)
{
  if(arg->opt == 0) return;
  ops_dat dat = arg->dat;
  sub_block_list sb = OPS_sub_block_list[dat->block->index];
  sub_dat_list sd = OPS_sub_dat_list[dat->index];
  int left_boundary_modified[OPS_MAX_DIM] = {0};
  int left_halo_modified[OPS_MAX_DIM] = {0};
  int right_boundary_modified[OPS_MAX_DIM] = {0};
  int right_halo_modified[OPS_MAX_DIM] = {0};

  int range_intersect[OPS_MAX_DIM] = {0};

  int ndim = sb->ndim;

  for (int dim = 0; dim < ndim; dim++) {
    range_intersect [dim] = intersection( iter_range[2*dim],
                                          iter_range[2*dim+1],
                                          sb->istart[dim],
                                          sb->iend[dim]+1); //i.e. the intersection of the execution range with my full range

    left_boundary_modified [dim] = intersection(iter_range[2*dim],
                                          iter_range[2*dim+1],
                                          sb->istart[dim],
                                          sb->istart[dim]+MAX_DEPTH-1); //i.e. the intersection of the execution range with my left boundary
    right_halo_modified[dim] = intersection(iter_range[2*dim],
                                          iter_range[2*dim+1],
                                          sb->iend[dim]+1,
                                          sb->iend[dim]+1+MAX_DEPTH-1); //i.e. the intersection of the execution range with the my right neighbour's boundary
    right_boundary_modified[dim] = intersection(iter_range[2*dim],
                                          iter_range[2*dim+1],
                                          sb->iend[dim]+1-MAX_DEPTH+1,
                                          sb->iend[dim]+1);
    left_halo_modified [dim] = intersection(iter_range[2*dim],
                                          iter_range[2*dim+1],
                                          sb->istart[dim]-MAX_DEPTH+1,
                                          sb->istart[dim]);
  }
  
  arg->dat->dirtybit = 1;
  for (int dim = 0; dim < ndim; dim++) {
    int other_dims = 1;
    for (int d2 = 0; d2 < ndim; d2++)
      if (d2 != dim) other_dims = other_dims && range_intersect[d2]>0;
    
    if (left_boundary_modified[dim]>0 && other_dims) {
      int beg = 1 + (iter_range[2*dim] >= sb->istart[dim] ? iter_range[2*dim] - sb->istart[dim] : 0);
      for (int d2 = beg; d2 < beg+left_boundary_modified[dim]; d2++) { //we shifted dirtybits, [1] is the first layer not the second
        arg->dat->dirty_dir_send[2*MAX_DEPTH*dim + d2] = 1;
      }
    }
    if (left_halo_modified[dim]>0 && other_dims) {
      int beg = iter_range[2*dim] >= sb->istart[dim]-MAX_DEPTH+1 ? iter_range[2*dim] - (sb->istart[dim] - MAX_DEPTH + 1) : 0;
      for (int d2 = beg; d2 < beg+left_halo_modified[dim]; d2++){
        arg->dat->dirty_dir_recv[2*MAX_DEPTH*dim + MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_boundary_modified[dim]>0 && other_dims) {
      int beg = iter_range[2*dim] >= sb->iend[dim]+1-MAX_DEPTH+1 ? iter_range[2*dim] - (sb->iend[dim]+1 - MAX_DEPTH + 1) : 0;
      for (int d2 = beg; d2 < beg+right_boundary_modified[dim]; d2++){
        arg->dat->dirty_dir_send[2*MAX_DEPTH*dim + 2*MAX_DEPTH - d2 - 1] = 1;
      }
    }
    if (right_halo_modified[dim]>0 && other_dims) {
      int beg = 1 + (iter_range[2*dim] >= sb->iend[dim]+1 ? iter_range[2*dim] - sb->iend[dim]+1 : 0);
      for (int d2 = beg; d2 < beg+right_halo_modified[dim]; d2++){
        arg->dat->dirty_dir_recv[2*MAX_DEPTH*dim + MAX_DEPTH + d2] = 1;
      }
    }
  }
}


void ops_H_D_exchanges(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}

void ops_set_dirtybit_host(ops_arg *args, int nargs)
{
  (void)nargs;
  (void)args;
}
