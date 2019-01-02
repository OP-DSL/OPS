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
 * @details Implementations of the core library functions utilized by all OPS
 * backends
 */
#include "ops_lib_core.h"
#include <stdlib.h>

#include <vector>

/////////////////////////////////////////////////////////////////////////
// Data structures
/////////////////////////////////////////////////////////////////////////

std::vector<ops_kernel_descriptor *> ops_kernel_dag;

/////////////////////////////////////////////////////////////////////////
// Enqueueing loops
// - add kernels to the DAG and execute immediately on CPU
/////////////////////////////////////////////////////////////////////////

void ops_add_to_dag(ops_kernel_descriptor *desc) {
  ops_kernel_dag.push_back(desc);
  
  // Run the kernel
  //desc->function(desc);
}

void ops_traverse_dag() {
  for(unsigned i = 0; i < ops_kernel_dag.size(); ++i){
    ops_printf("rerun %s\n", ops_kernel_dag[i]->name);
    ops_kernel_dag[i]->function(ops_kernel_dag[i]); 
  }
}


