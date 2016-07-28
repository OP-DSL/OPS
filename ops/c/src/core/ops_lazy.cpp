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
#include <sys/time.h>

#include <vector>
using namespace std;

std::vector<ops_kernel_descriptor *> ops_kernel_list;

void ops_execute();

void ops_enqueue_kernel(ops_kernel_descriptor *desc) {
  ops_kernel_list.push_back(desc);
  for (int i = 0; i < desc->nargs; i++) {
    if (desc->args[i].argtype == OPS_ARG_GBL && desc->args[i].acc != OPS_READ) {
      //      printf("Triggering execution at %s\n", desc->name);
      ops_execute();
      break;
    }
  }
}

void ops_execute() {
  for (unsigned int i = 0; i < ops_kernel_list.size(); i++) {
    // printf("Executing %s\n",ops_kernel_list[i]->name);
    ops_kernel_list[i]->function(ops_kernel_list[i]);
    free(ops_kernel_list[i]->args);
    free(ops_kernel_list[i]);
  }
  ops_kernel_list.clear();
}
