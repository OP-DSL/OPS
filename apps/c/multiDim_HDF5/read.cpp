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

/** @Application for testing muti-dimension ops_dats being read from HDF5 files
  * @author Piotr Zacharzewski and Gihan Mudalige
  */

#define OPS_3D
#include "ops_seq_v2.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {

  //*******************************************************************
  // INITIALISE OPS
  //---------------------------------------
  ops_init(argc, argv, 5);
  ops_printf("Hello world from OPS!\n\n");
  //*******************************************************************

  // THIS IS THE MAIN DIFFERENCE BETWEEN THIS AND THE OTHER PROGRAM
  ops_block block = ops_decl_block_hdf5(3, "grid0", "write_data.h5");

  ops_dat single =
      ops_decl_dat_hdf5(block, 1, "double", "single", "write_data.h5");
  ops_dat multi =
      ops_decl_dat_hdf5(block, 2, "double", "multi", "write_data.h5");
  ops_dat integ = ops_decl_dat_hdf5(block, 1, "int", "integ", "write_data.h5");

  ops_partition("empty_string_that_does_nothing_yet");
  ops_diagnostic_output();

  ops_fetch_block_hdf5_file(block, "read_data.h5");
  ops_fetch_dat_hdf5_file(multi, "read_data.h5");
  ops_fetch_dat_hdf5_file(single, "read_data.h5");
  ops_fetch_dat_hdf5_file(integ, "read_data.h5");

  int my_const;
  ops_get_const_hdf5("my_const", 1, "int", (char*)&my_const, "write_data.h5");
  printf("Read const: %d\n", my_const);

  char buffer[50];
  ops_get_const_hdf5("my_text", 11, "char", buffer, "write_data.h5");
  printf("Read text: %s\n", buffer);

  ops_write_const_hdf5("my_const", 1, "int", (char*)&my_const, "read_data.h5");
  ops_write_const_hdf5("my_text", 11, "char", (char*)buffer, "read_data.h5");


  //*******************************************************************
  // EXIT OPS AND PRINT TIMING INFO
  //---------------------------------------
  ops_timing_output(std::cout);
  ops_printf("\nSucessful exit from OPS!\n");
  ops_exit();
  //*******************************************************************
}
