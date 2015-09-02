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
* before writing to HDF5 files
*******************************************************************************/
void remove_mpi_halos(ops_dat dat, hsize_t* size, hsize_t* disp, char* data){

  int index = 0; int count = 0;
  //for(int m = disp[4]; m < size[4]; m++) {
  //  for(int l = disp[3]; l < size[3]; l++) {
   //   for(int k = disp[2]; k < size[2]; k++) {
        for(int j = disp[1]; j < size[1]; j++) {
          for(int i = disp[0]; i < size[0]; i++) {
              index = i +
                      j * size[0] ;//+
                      //k * size[0] * size[1] +
                     // l * size[0] * size[1] * size[2] +
                      //m * size[0] * size[1] * size[2] * size[3];
                      //data_d[count] = *((double *)&(dat->data[index]));
              memcpy(&data[count*dat->elem_size],&dat->data[index*dat->elem_size],dat->elem_size);
              count++;
              //printf("index %d, count %d\n",index,count);
          }
        }
     //}
    //}
  //}


  printf("index %d, count %d\n",index,count);
  return ;//data;
  /*
  x = disp[0] to size[0]
  y = disp[1] to size[1]
  z = disp[2] to size[2]
  t = disp[3] to size[4]
  u = disp[4] to size[4]
  index = x + y * D1 + z * D1 * D2 + t * D1 * D2 * D3 + u * D1 * D2 * D3 * D4;
  */

}

/*******************************************************************************
* Routine to write an ops_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set does not exists in file creates data set
*******************************************************************************/

void ops_fetch_data_hdf5_file(ops_dat dat, char const *file_name) {

  //fetch data onto the host ( if needed ) based on the backend


  //complute the number of elements that this process will write to the final file
  //also compute the correct offsets on the final file that this process should begin from to write
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  ops_block block = dat->block;
  sub_block *sb = OPS_sub_block_list[dat->block->index];

  hsize_t disp[block->dims]; //temp array to hold disps of the dat to write to hdf5 file
  hsize_t size[block->dims]; //temp array to hold size of the dat to write to hdf5 file
  hsize_t g_size[block->dims]; //temp array to hold global size of the dat to write to hdf5 file
  int g_d_m[block->dims]; //temp array to hold global size of the block halo (-) depth to write to hdf5 file
  int g_d_p[block->dims]; //temp array to hold global size of the block halo (+) depth to write to hdf5 file
  for (int d = 0; d < block->dims; d++){
    // remove left MPI halo to get start disp from begining of dat
    // include left block halo
    disp[d] = sd->decomp_disp[d] - sd->d_im[d] - dat->d_m[d];
    size[d] = sd->decomp_size[d]; //local size to be written to hdf5 file
    g_size[d] = sd->gbl_size[d]; //global size to be written to hdf5 file
    g_d_m[d] = sd->gbl_d_m[d]; //global halo depth(-) to be written to hdf5 file
    g_d_p[d] = sd->gbl_d_p[d]; //global halo depth(+) to be written to hdf5 file
    printf("disp[%d] = %d ",d,disp[d]);
    printf("size[%d] = %d ",d,size[d]);
    printf("gbl_size[%d] = %d ",d,g_size[d]);
    printf("dat->d_m[%d] = %d ",d,g_d_m[d]);
    printf("dat->d_p[%d] = %d ",d,g_d_p[d]);
  }
  printf("\n");

  int t_size = 1;
  for (int d = 0; d < dat->block->dims; d++) t_size *= size[d];
  printf("t_size = %d ",t_size);
  char* data = (char *)malloc(t_size*dat->elem_size);
  double* data_d = (double *) data;

  remove_mpi_halos(dat, size, disp, data);
  for (int t = 0; t<t_size; t++) printf("%lf \n",data_d[t]);
  printf("****************************\n\n");
  //free(data); need to after writing to hdf5 file


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
  hid_t dataspace;    //data space identifier
  hid_t plist_id;     //property list identifier
  hid_t memspace;     //memory space identifier
  hid_t attr;         //attribute identifier
  herr_t err;         //error code

  //Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OPS_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {

    ops_printf("File %s does not exist .... creating file\n", file_name);

    //create file
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    if (ops_is_root()) {
      FILE *fp; fp = fopen(file_name, "w");
      fclose(fp);
    }
    MPI_Barrier(OPS_MPI_HDF5_WORLD);
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  //file exists
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);

  //Each process defines dataset in memory and writes it to a hyperslab in the file.
  memspace = H5Screate_simple(block->dims, size, NULL);

  if(H5Lexists(file_id, dat->name, H5P_DEFAULT) == 0) {
    //data set does not exist .. create it
    ops_printf("ops_dat %s does not exists in the file ... creating data set\n", dat->name);

    //Create the dataspace for the dataset
    dataspace = H5Screate_simple(block->dims, g_size, NULL);
    //Select hyperslab in the file.
    err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, disp, NULL, size, NULL);
    H5Pclose(plist_id);
    //Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    //Create the dataset with default properties and close dataspace.
    if(strcmp(dat->type,"double") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_DOUBLE, dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    else if(strcmp(dat->type,"float") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_FLOAT, dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    else if(strcmp(dat->type,"int") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_INT, dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    else if(strcmp(dat->type,"long") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LONG, dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    else if(strcmp(dat->type,"long long") == 0)
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LLONG, dataspace,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    else {
      printf("Unknown type in ops_fetch_data_hdf5_file()\n");
      MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
    }
    H5Dclose(dset_id);

    //attach attributes to dat
    H5LTset_attribute_string(file_id, dat->name, "block", block->name); //block
    H5LTset_attribute_int(file_id, dat->name, "dim", &(dat->dim), 1); //dim
    H5LTset_attribute_int(file_id, dat->name, "size", sd->gbl_size, block->dims); //size
    H5LTset_attribute_int(file_id, dat->name, "d_m", g_d_m, block->dims); //d_m
    H5LTset_attribute_int(file_id, dat->name, "d_p", g_d_p, block->dims); //d_p
    H5LTset_attribute_string(file_id, dat->name, "type", dat->type); //type


  }

  //update data existing or created data set
  dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);
  //Get the dataspace for the dataset
  dataspace = H5Dget_space(dset_id);
  //Select hyperslab
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, disp, NULL, size, NULL);
  H5Pclose(plist_id);
  //Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  //write data
  if(strcmp(dat->type,"double") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist_id, data);
  else if(strcmp(dat->type,"float") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, data);
  else if(strcmp(dat->type,"int") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id, data);
  else if(strcmp(dat->type,"long") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id, data);
  else if(strcmp(dat->type,"long long") == 0)
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id, data);
  else {
    printf("Unknown type in ops_fetch_data_hdf5_file()\n");
    MPI_Abort(OPS_MPI_HDF5_WORLD, 2);
  }

  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Fclose(file_id);
  MPI_Comm_free(&OPS_MPI_HDF5_WORLD);
  free(data);
  return;
}