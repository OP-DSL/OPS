#ifndef __OPS_MPI_CORE_H
#define __OPS_MPI_CORE_H
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

/** @file
  * @brief core header file for the ops MPI backend
  * @author Gihan Mudalige, Istvan Reguly
  * @details Header file for OPS MPI backend
  */
#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <mpi.h>
#include <ops_lib_core.h>

/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0

#ifdef __cplusplus
extern "C" {
#endif

///
/// Struct for holding the decomposition details of a block on an MPI process
///
typedef struct {
  /// the decomposition is for this block
  ops_block block;
  /// number of dimensions;
  int ndim;
  /// number of processors in each dimension
  int *pdims;
  /// my MPI rank in each dimension (in cart cords)
  int coords[OPS_MAX_DIM];
  /// previous neighbor in each dimension (in cart cords)
  int id_m[OPS_MAX_DIM];
  /// next neighbor in each dimension (in cart cords)
  int id_p[OPS_MAX_DIM];
  /// finest level decomposed details
  int decomp_disp[OPS_MAX_DIM];
  int decomp_size[OPS_MAX_DIM];
  /// normal communicator for intra-block
  MPI_Comm comm1;
  /// Cartesian communicator for intra-block
  MPI_Comm comm;
  /// Group communicator for intra-block
  MPI_Group grp;
  int owned;
} sub_block;

typedef sub_block *sub_block_list;

///
/// Struct duplicating information in MPI_Datatypes for (strided) halo access
///

typedef struct {
  int count;       ///< number of blocks
  int blocklength; ///< size of blocks
  int stride;      ///< stride between blocks
} ops_int_halo;

///
/// Struct for holding the decomposition details of a dat on an MPI process
///

typedef struct {
  /// the decomposition is for this dat
  ops_dat dat;
  /// product array -- used for MPI send/Receives
  int *prod;
  /// MPI Types for send/receive -- these should be defined for the dat, not the
  /// block
  MPI_Datatype *mpidat;
  /// data structures describing halo access
  ops_int_halo *halos;
  /// the size of the local sub-block in each dimension, "owned"
  int decomp_size[OPS_MAX_DIM];
  /// the displacement from the start of the block in each dimension
  int decomp_disp[OPS_MAX_DIM];

  /// intra-block halo
  int d_im[OPS_MAX_DIM];
  int d_ip[OPS_MAX_DIM];

  /// global information
  int gbl_size[OPS_MAX_DIM];
  int gbl_base[OPS_MAX_DIM];
  int gbl_d_m[OPS_MAX_DIM];
  int gbl_d_p[OPS_MAX_DIM];

  /// flag to indicate MPI halo exchange is needed
  int dirtybit;
  /// flag to indicate MPI halo exchange in a direction is needed
  int *dirty_dir_send;
  /// flag to indicate MPI halo exchange in a direction is needed
  int *dirty_dir_recv;

} sub_dat;

typedef sub_dat *sub_dat_list;

typedef struct {
  ops_halo halo;
  int nproc_from; ///< number of processes I have to send to (from part of halo)
  int nproc_to;///< number of processes I have to receive from (to part of halo)
  int *proclist;
  int *local_from_base;
  int *local_to_base;
  int *local_iter_size;
  int index;
} ops_mpi_halo;

typedef struct {
  ops_halo_group group;
  int nhalos;
  ops_mpi_halo **mpi_halos;
  int index;
  int num_neighbors_send;
  int num_neighbors_recv;
  int *neighbors_send;
  int *neighbors_recv;
  int *send_sizes;
  int *recv_sizes;
  MPI_Request *requests;
  MPI_Status *statuses;
} ops_mpi_halo_group;

void ops_mpi_exit();

/*******************************************************************************
* External functions defined in ops_mpi_(cuda)_rt_support.c
*******************************************************************************/
void ops_pack(ops_dat dat, const int src_offset, char *__restrict dest,
              const ops_int_halo *__restrict halo);
void ops_unpack(ops_dat dat, const int dest_offset, const char *__restrict src,
                const ops_int_halo *__restrict halo);
char* OPS_realloc_fast(char *ptr, size_t old_size, size_t new_size);

/*******************************************************************************
* Other External functions
*******************************************************************************/


#ifdef __cplusplus
}
#endif

//
// MPI Communicator for halo creation and exchange
//

extern MPI_Comm OPS_MPI_GLOBAL;
extern int ops_comm_global_size;
extern int ops_my_global_rank;

//
// list holding sub-block and sub-dat geometries
//
extern sub_block_list *OPS_sub_block_list;
extern sub_dat_list *OPS_sub_dat_list;
extern ops_mpi_halo *OPS_mpi_halo_list;
extern ops_mpi_halo_group *OPS_mpi_halo_group_list;

  
extern double ops_gather_time;
extern double ops_scatter_time;
extern double ops_sendrecv_time;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
#endif /*__OPS_MPI_CORE_H*/
