/* Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/** @brief Mesh chunk generation driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invoked the users specified chunk generator.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file

#define OPS_3D
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "generate_chunk_kernel.h"


void generate()
{
  /** This function is left empty as we read the values of
  density0, energy0, xvel0, yvel0 and zvel0 from the cloverdata.h5 HDF5 file
  **/

  /** wirte back the dats to a test file to
  compare agains the original**/
    //ops_fetch_dat_hdf5_file(density0, "test_cloverdata.h5");
    //ops_fetch_dat_hdf5_file(energy0, "test_cloverdata.h5");
    //ops_fetch_dat_hdf5_file(xvel0, "test_cloverdata.h5");
    //ops_fetch_dat_hdf5_file(yvel0, "test_cloverdata.h5");
    //ops_fetch_dat_hdf5_file(zvel0, "test_cloverdata.h5");
    //ops_fetch_block_hdf5_file(clover_grid, "cloverdata.h5");
}