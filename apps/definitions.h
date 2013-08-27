/*Crown Copyright 2012 AWE.

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

/** @brief Holds the high level data types
 *  @author Wayne Gaudin
 *  @details The high level data types used to store the mesh and field data
 *  are defined here.

 *  Also the global variables used for defining the input and controlling the
 *  scheme are defined here.
*/

#ifndef __CLOVER_LEAF_DEFINITIONS_H
#define __CLOVER_LEAF_DEFINITIONS_H

#define FALSE 0
#define TRUE 1

typedef struct
{
      int defined;  //logical
      double density,
             energy,
             xvel,
             yvel;
      int geometry;
      double xmin,
             xmax,
             ymin,
             ymax,
             radius;
} state_type_core;
typedef state_type_core * state_type;

extern state_type * states;

#endif /* __CLOVER_LEAF_DEFINITIONS_H */
