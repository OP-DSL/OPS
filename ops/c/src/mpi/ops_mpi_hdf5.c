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

/** @brief HDF5 file I/O backend implementation for MPI
  * @author Gihan Mudalige (started 28-08-2015)
  * @details Implements the OPS API calls for the HDF5 file I/O functionality
  */

#include <mpi.h>
#include <ops_mpi_core.h>
#include <math.h>
#include <ops_util.h>


// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

//hdf5 header
#include <hdf5.h>
#include <hdf5_hl.h>

//
//MPI Communicator for parallel I/O
//

MPI_Comm OPS_MPI_HDF5_WORLD;

sub_block_list *OPS_sub_block_list;// pointer to list holding sub-block
                                   // geometries


sub_dat_list *OPS_sub_dat_list;// pointer to list holding sub-dat
                                 // details

/*******************************************************************************
* Routine to remove the intra-block halos from the flattend 1D dat
* before writing to HDF5 files - Maximum dimension of block is 2
*******************************************************************************/
void remove_mpi_halos2D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){

  int index = 0; int count = 0;
  //for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  //  for(int k = disp[2]; k < disp[2]+size[2]; k++) {
    for(int j = disp[1]; j < disp[1]+size[1]; j++) {
      for(int i = disp[0]; i < disp[0]+size[0]; i++) {
          index = i +
                  j * dat->size[0]; //+ // need to stride in dat->size as data block includes intra-block halos
                  //k * dat->size[0] * dat->size[1];// +
                  //l * dat->size[0] * dat->size[1] * dat->size[2] +
                  //m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
          memcpy(&data[count*dat->elem_size],&dat->data[index*dat->elem_size],dat->elem_size);
          count++;
      }
    }
  //}
  //}
  //}


  return;
  /*
  x = disp[0] to size[0]
  y = disp[1] to size[1]
  z = disp[2] to size[2]
  t = disp[3] to size[4]
  u = disp[4] to size[4]
  index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3 + u * D1 * D2 * D3 * D4;


  D1 - dat->size[0]
  D2 - dat->size[1]
  D3 - dat->size[2]
  D4 - dat->size[3]
  */

}

/*******************************************************************************
* Routine to remove the intra-block halos from the flattend 1D dat
* before writing to HDF5 files - Maximum dimension of block is 4
*******************************************************************************/
void remove_mpi_halos3D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){
}

/*******************************************************************************
* Routine to remove the intra-block halos from the flattend 1D dat
* before writing to HDF5 files - Maximum dimension of block is 4
*******************************************************************************/
void remove_mpi_halos4D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){
}

/*******************************************************************************
* Routine to remove the intra-block halos from the flattend 1D dat
* before writing to HDF5 files - Maximum dimension of block is 5
*******************************************************************************/
void remove_mpi_halos5D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){
}



/*******************************************************************************
* Routine to add the intra-block halos from the flattend 1D data
* after reading from an HDF5 file - Maximum dimension of block is 2
*******************************************************************************/
void add_mpi_halos2D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){
  int index = 0; int count = 0;
  //for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
  //  for(int k = disp[2]; k < disp[2]+size[2]; k++) {
    for(int j = disp[1]; j < disp[1]+size[1]; j++) {
      for(int i = disp[0]; i < disp[0]+size[0]; i++) {
          index = i +
                  j * dat->size[0]; //+ // need to stride in dat->size as data block includes intra-block halos
                  //k * dat->size[0] * dat->size[1];// +
                  //l * dat->size[0] * dat->size[1] * dat->size[2] +
                  //m * dat->size[0] * dat->size[1] * dat->size[2] * dat->size[3];
          memcpy(&dat->data[index*dat->elem_size],&data[count*dat->elem_size],dat->elem_size);
          count++;
      }
    }
  //}
  //}
  //}
  return;
}

void add_mpi_halos3D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){};
void add_mpi_halos4D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){};
void add_mpi_halos5D(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){};

/*******************************************************************************
* Routine to write an ops_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set does not exists in file creates data set
*******************************************************************************/

void ops_fetch_data_hdf5_file(ops_dat dat, char const *file_name) {

  //fetch data onto the host ( if needed ) based on the backend -- TODO for GPUs

  //complute the number of elements that this process will write to the final file
  //also compute the correct offsets on the final file that this process should begin from to write
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  ops_block block = dat->block;
  sub_block *sb = OPS_sub_block_list[dat->block->index];

  hsize_t disp[block->dims]; //global disps to compute the chunk data set dimensions
  hsize_t l_disp[block->dims]; //local disps to remove MPI halos
  hsize_t size[block->dims]; //local size to compute the chunk data set dimensions
  hsize_t gbl_size[block->dims]; //global size to compute the chunk data set dimensions

  int g_size[block->dims]; //global size of the dat attribute to write to hdf5 file
  int g_d_m[block->dims]; //global size of the block halo (-) depth attribute to write to hdf5 file
  int g_d_p[block->dims]; //global size of the block halo (+) depth attribute to write to hdf5 file

  hsize_t count[block->dims]; //parameters for for hdf5 file chuck writing
  hsize_t stride[block->dims]; //parameters for for hdf5 file chuck writing

  for (int d = 0; d < block->dims; d++){
    // remove left MPI halo to get start disp from begining of dat
    // include left block halo
    disp[d] = sd->decomp_disp[d] - sd->d_im[d] - dat->d_m[d]; //global displacements of the data set
    l_disp[d] = 0 - sd->d_im[d]; //local displacements of the data set (i.e. per MPI proc)
    size[d] = sd->decomp_size[d]; //local size to compute the chunk data set dimensions
    gbl_size[d] = sd->gbl_size[d]; //global size to compute the chunk data set dimensions

    g_d_m[d] = sd->gbl_d_m[d]; //global halo depth(-) attribute to be written to hdf5 file
    g_d_p[d] = sd->gbl_d_p[d]; //global halo depth(+) attribute to be written to hdf5 file
    g_size[d] = sd->gbl_size[d] + g_d_m[d] - g_d_p[d]; //global size attribute to be written to hdf5 file

    count[d] = 1; stride[d] = 1;

    //printf("l_disp[%d] = %d ",d,l_disp[d]);
    //printf("disp[%d] = %d ",d,disp[d]);
    //printf("size[%d] = %d ",d,size[d]);
    //printf("dat->size[%d] = %d ",d,dat->size[d]);
    //printf("gbl_size[%d] = %d ",d,sd->gbl_size[d]);
    //printf("g_size[%d] = %d ",d,g_size[d]);
    //printf("dat->d_m[%d] = %d ",d,g_d_m[d]);
    //printf("dat->d_p[%d] = %d ",d,g_d_p[d]);
  }

  int t_size = 1;
  for (int d = 0; d < dat->block->dims; d++) t_size *= size[d];
  //printf("t_size = %d ",t_size);
  char* data = (char *)malloc(t_size*dat->elem_size);

   //create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  if(block->dims == 2)
    remove_mpi_halos2D(dat, size, l_disp, data);
  else if(block->dims == 3)
    remove_mpi_halos3D(dat, size, l_disp, data);
  else if (block->dims == 4)
    remove_mpi_halos4D(dat, size, l_disp, data);
  else if (block->dims == 5)
    remove_mpi_halos5D(dat, size, l_disp, data);

  //MPI variables
  MPI_Info info  = MPI_INFO_NULL;

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    ops_printf("File %s does not exist .... creating file\n", file_name);
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    if (ops_is_root()) {
      FILE *fp; fp = fopen(file_name, "w");
      fclose(fp);
    }
    //Create a new file collectively and release property list identifier.
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);
  if(H5Lexists(file_id, dat->name, H5P_DEFAULT) == 0) {
    ops_printf("ops_dat %s does not exists in the file ... creating data\n", dat->name);

    //Create the dataspace for the dataset
    filespace = H5Screate_simple(block->dims, gbl_size, NULL); //space in file

    // Create chunked dataset
    plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, block->dims, gbl_size); //chunk data set need to be the same size on each proc

    //Create the dataset with default properties and close filespace.
    if(strcmp(dat->type,"double") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_DOUBLE, filespace,
          H5P_DEFAULT, plist_id, H5P_DEFAULT);
    else if(strcmp(dat->type,"float") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_FLOAT, filespace,
          H5P_DEFAULT, plist_id, H5P_DEFAULT);
    else if(strcmp(dat->type,"int") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_INT, filespace,
          H5P_DEFAULT, plist_id, H5P_DEFAULT);
    else if(strcmp(dat->type,"long") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LONG, filespace,
          H5P_DEFAULT, plist_id, H5P_DEFAULT);
    else if(strcmp(dat->type,"long long") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LLONG, filespace,
          H5P_DEFAULT, plist_id, H5P_DEFAULT);
    else {
      printf("Unknown type in ops_fetch_data_hdf5_file()\n");
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Dclose(dset_id);

    //attach attributes to dat
    H5LTset_attribute_string(file_id, dat->name, "block", block->name); //block
    H5LTset_attribute_int(file_id, dat->name, "block_index", &(block->index), 1); //block index
    H5LTset_attribute_int(file_id, dat->name, "dim", &(dat->dim), 1); //dim
    H5LTset_attribute_int(file_id, dat->name, "size", g_size, block->dims); //size
    H5LTset_attribute_int(file_id, dat->name, "d_m", g_d_m, block->dims); //d_m
    H5LTset_attribute_int(file_id, dat->name, "d_p", g_d_p, block->dims); //d_p
    H5LTset_attribute_int(file_id, dat->name, "base", dat->base, block->dims); //base
    H5LTset_attribute_string(file_id, dat->name, "type", dat->type); //type
  }

  //open existing dat
  dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);

  //
  //check attributes .. error if not equal
  //
  char read_block_name[30];
  if (H5LTget_attribute_string(file_id, dat->name, "block", read_block_name) < 0){
    ops_printf("Attribute \"block\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  } else {
    if (strcmp(block->name,read_block_name) != 0) {
      ops_printf("BLocks on which data set %s is defined are not equal .. Aborting\n",dat->name);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }

  int read_block_index;
  if (H5LTget_attribute_int(file_id, dat->name, "block_index", &read_block_index) < 0) {
    ops_printf("Attribute \"block_index\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    if (block->index != read_block_index) {
      ops_printf("Unequal dims of data set %s: block index on file %d, block index to be wirtten %d .. Aborting\n",
         dat->name,read_block_index, block->index);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }

  int read_dim;
  if (H5LTget_attribute_int(file_id, dat->name, "dim", &read_dim) < 0) {
    ops_printf("Attribute \"dim\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    if (dat->dim != read_dim) {
      ops_printf("Unequal dims of data set %s: dim on file %d, dim to be wirtten %d .. Aborting\n",
         dat->name,read_dim, dat->dim);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }

  int read_size[block->dims];
  if (H5LTget_attribute_int(file_id, dat->name, "size", read_size) < 0) {
    ops_printf("Attribute \"size\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    for(int d = 0; d<block->dims; d++) {
      if (g_size[d] != read_size[d]) {
        ops_printf("Unequal sizes of data set %s: size[%d] on file %d, size[%d] to be wirtten %d .. Aborting\n",
           dat->name, d, read_size[d], d, g_size[d]);
        MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
      }
    }
  }

  int read_d_m[block->dims];
  if (H5LTget_attribute_int(file_id, dat->name, "d_m", read_d_m) < 0) {
    ops_printf("Attribute \"d_m\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    for(int d = 0; d<block->dims; d++) {
      if (g_d_m[d] != read_d_m[d]) {
        ops_printf("Unequal d_m of data set %s: g_d_m[%d] on file %d, g_d_m[%d] to be wirtten %d .. Aborting\n",
           dat->name, d, read_d_m[d], d, g_d_m[d]);
        MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
      }
    }
  }

  int read_d_p[block->dims];
  if (H5LTget_attribute_int(file_id, dat->name, "d_p", read_d_p) < 0) {
    ops_printf("Attribute \"d_p\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    for(int d = 0; d<block->dims; d++) {
      if (g_d_p[d] != read_d_p[d]) {
        ops_printf("Unequal d_p of data set %s: g_d_p[%d] on file %d, g_d_p[%d] to be wirtten %d .. Aborting\n",
           dat->name, d, read_d_p[d], d, g_d_p[d]);
        MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
      }
    }
  }

  int read_base[block->dims];
  if (H5LTget_attribute_int(file_id, dat->name, "base", read_base) < 0) {
    ops_printf("Attribute \"base\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    for(int d = 0; d<block->dims; d++) {
      if (dat->base[d] != read_base[d]) {
        ops_printf("Unequal base of data set %s: base[%d] on file %d, base[%d] to be wirtten %d .. Aborting\n",
           dat->name, d, read_base[d], d, dat->base[d]);
        MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
      }
    }
  }

  char read_type[15];
  if (H5LTget_attribute_string(file_id, dat->name, "type", read_type) < 0){
    ops_printf("Attribute \"type\" not found in data set %s .. Aborting\n",dat->name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  } else {
    if (strcmp(dat->type,read_type) != 0) {
      ops_printf("Type of data of data set %s is not equal: type on file %s, type specified %s .. Aborting\n",
        dat->name, read_type, dat->type);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }

  //Need to flip the dimensions to accurately write to HDF5 chunk decomposition
  hsize_t DISP[block->dims];
  DISP[0] = disp[1];
  DISP[1] = disp[0];
  hsize_t SIZE[block->dims];
  SIZE[0] = size[1];
  SIZE[1] = size[0];

  memspace = H5Screate_simple(block->dims, size, NULL); //block of memory to write to file by each proc

  //Select hyperslab
  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, DISP, stride, count, SIZE);

  //Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  //write data
  if(strcmp(dat->type,"double") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"float") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"int") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"long") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"long long") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, filespace, plist_id, data);
  else {
    printf("Unknown type in ops_fetch_data_hdf5_file()\n");
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  free(data);

  H5Sclose(filespace);
  H5Pclose(plist_id);
  H5Dclose(dset_id);
  H5Sclose(memspace);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);

  return;
}

 ops_dat ops_decl_dat_hdf5(ops_block block, int dat_size,
                      char const *type,
                      char const *dat_name,
                      char const *file_name) {

  //create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  //MPI variables
  MPI_Info info  = MPI_INFO_NULL;

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //open given hdf5 file .. if it exists
  if (file_exist(file_name) == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    ops_printf("File %s does not exist .... aborting\n", file_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  //check if ops_dat exists
  if(H5Lexists(file_id, dat_name, H5P_DEFAULT) == 0)
    ops_printf("ops_dat %s does not exists in the file ... aborting\n", dat_name);

  //ops_dat exists .. now check block_index, type and dim
  int read_block_index;
  if (H5LTget_attribute_int(file_id, dat_name, "block_index", &read_block_index) < 0) {
    ops_printf("Attribute \"block_index\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    if (block->index != read_block_index) {
      ops_printf("Unequal dims of data set %s: block index on file %d, block index specified for this dat %d .. Aborting\n",
         dat_name,read_block_index, block->index);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }
  int read_dim;
  if (H5LTget_attribute_int(file_id, dat_name, "dim", &read_dim) < 0) {
    ops_printf("Attribute \"dim\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }
  else {
    if (dat_size != read_dim) {
      ops_printf("Unequal dims of data set %s: dim on file %d, dim specified %d .. Aborting\n",
         dat_name,read_dim, dat_size);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }
  char read_type[7];
  if (H5LTget_attribute_string(file_id, dat_name, "type", read_type) < 0){
    ops_printf("Attribute \"type\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  } else {
    if (strcmp(type,read_type) != 0) {
      ops_printf("Type of data of data set %s is not equal: type on file %s, type specified %s .. Aborting\n",
        dat_name, read_type, type);
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
  }

  //checks passed .. now read in all other details of ops_dat from file

  int read_size[block->dims];
  if (H5LTget_attribute_int(file_id, dat_name, "size", read_size) < 0) {
    ops_printf("Attribute \"size\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  int read_d_m[block->dims];
  if (H5LTget_attribute_int(file_id, dat_name, "d_m", read_d_m) < 0) {
    ops_printf("Attribute \"d_m\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  int read_d_p[block->dims];
  if (H5LTget_attribute_int(file_id, dat_name, "d_p", read_d_p) < 0) {
    ops_printf("Attribute \"d_p\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  int read_base[block->dims];
  if (H5LTget_attribute_int(file_id, dat_name, "base", read_base) < 0) {
    ops_printf("Attribute \"base\" not found in data set %s .. Aborting\n",dat_name);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  //set type size
  int type_size;
  if(strcmp(read_type,"double") == 0)
    type_size = sizeof(double);
  else if(strcmp(read_type,"float") == 0)
    type_size = sizeof(float);
  else if(strcmp(read_type,"int") == 0)
    type_size = sizeof(int);
  else if(strcmp(read_type,"long") == 0)
    type_size = sizeof(long);
  else if(strcmp(read_type,"long long") == 0)
    type_size = sizeof(long long);
  else{
    printf("Unknown type %s in ops_decl_dat_hdf5()\n", read_type);
    exit(2);
  }

  char * data = NULL;

  ops_dat created_dat = ops_decl_dat_char(block, dat_size,
      read_size/*global dat size in each dimension*/,
      read_base, read_d_m, read_d_p, data/*null for now*/,
      type_size/*size of(type)*/, type, dat_name );

  created_dat->is_hdf5 = 1;
  created_dat->hdf5_file = file_name;
  created_dat->user_managed = 0;

  H5Pclose(plist_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);

  return created_dat;
  /**
  When ops_decomp_dats are encountered read the correct hyperslab chunk form
  hdf5 file and pad the data with the correct mpi-halo depths
  then attache it to the ops_dat->data of this ops_dat
  **/
 }


void ops_read_dat_hdf5(ops_dat dat) {
  //complute the number of elements that this process will read from file
  //also compute the correct offsets on the file that this process should begin from to read
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  ops_block block = dat->block;
  sub_block *sb = OPS_sub_block_list[dat->block->index];

  hsize_t disp[block->dims]; //global disps to compute the chunk data set dimensions
  hsize_t l_disp[block->dims]; //local disps to remove MPI halos
  hsize_t size[block->dims]; //local size to compute the chunk data set dimensions
  hsize_t gbl_size[block->dims]; //global size to compute the chunk data set dimensions

  int g_d_m[block->dims]; //global size of the block halo (-) depth attribute to read from hdf5 file
  int g_d_p[block->dims]; //global size of the block halo (+) depth attribute to read from hdf5 file

  hsize_t count[block->dims]; //parameters for for hdf5 file chuck reading
  hsize_t stride[block->dims]; //parameters for for hdf5 file chuck reading

  for (int d = 0; d < block->dims; d++){
    // remove left MPI halo to get start disp from begining of dat
    // include left block halo
    disp[d] = sd->decomp_disp[d] - sd->d_im[d] - dat->d_m[d]; //global displacements of the data set
    l_disp[d] = 0 - sd->d_im[d]; //local displacements of the data set (i.e. per MPI proc)
    size[d] = sd->decomp_size[d]; //local size to compute the chunk data set dimensions
    gbl_size[d] = sd->gbl_size[d]; //global size to compute the chunk data set dimensions

    g_d_m[d] = sd->gbl_d_m[d]; //global halo depth(-) to be read from hdf5 file
    g_d_p[d] = sd->gbl_d_p[d]; //global halo depth(+) to be read from hdf5 file

    count[d] = 1; stride[d] = 1;

    //printf("l_disp[%d] = %d ",d,l_disp[d]);
    //printf("disp[%d] = %d ",d,disp[d]);
    //printf("size[%d] = %d ",d,size[d]);
    //printf("dat->size[%d] = %d ",d,dat->size[d]);
    //printf("gbl_size[%d] = %d ",d,gbl_size[d]);
    //printf("dat->d_m[%d] = %d ",d,g_d_m[d]);
    //printf("dat->d_p[%d] = %d ",d,g_d_p[d]);
  }

  int t_size = 1;
  for (int d = 0; d < dat->block->dims; d++) t_size *= size[d];
  //printf("t_size = %d ",t_size);
  char* data = (char *)malloc(t_size*dat->elem_size);


  //create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(MPI_COMM_WORLD, &OPS_MPI_HDF5_WORLD);
  MPI_Comm_rank(OPS_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OPS_MPI_HDF5_WORLD, &comm_size);

  //MPI variables
  MPI_Info info  = MPI_INFO_NULL;

  //HDF5 APIs definitions
  hid_t file_id;      //file identifier
  hid_t dset_id;      //dataset identifier
  hid_t filespace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //open given hdf5 file .. if it exists
  if (file_exist(dat->hdf5_file) == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    ops_printf("File %s does not exist .... aborting\n", dat->hdf5_file);
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);
  file_id = H5Fopen(dat->hdf5_file, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  //check if ops_dat exists
  if(H5Lexists(file_id, dat->name, H5P_DEFAULT) == 0)
    ops_printf("ops_dat %s does not exists in the file ... aborting\n", dat->name);

  dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);

  // Create chunked dataset
  plist_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist_id, block->dims, gbl_size); //chunk data set need to be the same size on each proc
  H5Pclose(plist_id);

  //Need to flip the dimensions to accurately read from HDF5 chunk decomposition
  hsize_t DISP[block->dims];
  DISP[0] = disp[1];
  DISP[1] = disp[0];
  hsize_t SIZE[block->dims];
  SIZE[0] = size[1];
  SIZE[1] = size[0];

  memspace = H5Screate_simple(block->dims, size, NULL); //block of memory to read from file by each proc

  //Select hyperslab
  filespace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, DISP, stride, count, SIZE);

  //Create property list for collective dataset read.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  //read data
  if(strcmp(dat->type,"double") == 0)
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"float") == 0)
    H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"int") == 0)
    H5Dread(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"long") == 0)
    H5Dread(dset_id, H5T_NATIVE_LONG, memspace, filespace, plist_id, data);
  else if(strcmp(dat->type,"long long") == 0)
    H5Dread(dset_id, H5T_NATIVE_LLONG, memspace, filespace, plist_id, data);
  else {
    printf("Unknown type in ops_fetch_data_hdf5_file()\n");
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  //add MPI halos
  if(block->dims == 2)
    add_mpi_halos2D(dat, size, l_disp, data);
  else if(block->dims == 3)
    add_mpi_halos3D(dat, size, l_disp, data);
  else if (block->dims == 4)
    add_mpi_halos4D(dat, size, l_disp, data);
  else if (block->dims == 5)
    add_mpi_halos5D(dat, size, l_disp, data);

  free(data);
  H5Sclose(filespace);
  H5Pclose(plist_id);
  H5Dclose(dset_id);
  H5Sclose(memspace);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);

  return;
}