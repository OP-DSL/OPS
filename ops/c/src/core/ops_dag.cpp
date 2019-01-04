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

#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <vector>

/////////////////////////////////////////////////////////////////////////
// Data structures
/////////////////////////////////////////////////////////////////////////

// TODO add get_reduction_result calls to this.
std::vector<ops_kernel_descriptor *>
    ops_kernel_dag; /**< Contains all called kernels in order. */

// Data structures to follow data changes

/* TODO(bgd54): not only indexes but the states of the data, and consider that
 * we write only a part of the data */
std::map<std::string, std::vector<int>>
    ops_dat_states; /**< For each ops_dat and reduction_handle contains all
                       kernel index where the data is changed. TODO update*/

std::map<std::string, char *>
    name2data; /**< just for checking if the names are unique as we hope */

/////////////////////////////////////////////////////////////////////////
// Enqueueing loops
// - add kernels to the DAG and execute immediately on CPU
/////////////////////////////////////////////////////////////////////////

void ops_add_to_dag(ops_kernel_descriptor *desc) {

  // register parameters to ops_dat_states
  for (int param_idx = 0; param_idx < desc->nargs; param_idx++) {
    ops_arg arg = desc->args[param_idx];
    if (arg.argtype == OPS_ARG_GBL) {
      continue; // TODO arg_gbl saves, reduction handle
    } else if (arg.argtype == OPS_ARG_DAT) {
      ops_dat dat = arg.dat;
      std::string name = dat->name;

      if (0 == name2data.count(name)) {
        name2data[name] = dat->data;
      } else {
        assert(name2data[name] == dat->data &&
               "name of an ops_dat must be unique\n");
      }

      if (arg.acc == OPS_READ && 0 == ops_dat_states.count(name)) {
        continue; // TODO ops_dat_states[dat].push_back(ops_kernel_dag.size());
                  // checkpointing
      } else if (arg.acc == OPS_WRITE) {
        ops_dat_states[name].push_back(ops_kernel_dag.size());
        // TODO firs appearence if we consider not whole ranges we need
        // checkpoint before and after
      } else if (arg.acc == OPS_INC) {
        ops_dat_states[name].push_back(ops_kernel_dag.size());
        // TODO we need checkpoint before and after
      } else if (arg.acc == OPS_RW) {
        ops_dat_states[name].push_back(ops_kernel_dag.size());
        // TODO we need checkpoint before and after
      }
    }
  }

  // Add desc to DAG
  ops_kernel_dag.push_back(desc);
  // Run the kernel
  desc->function(
      desc); // TODO we need to prepare it like in ops_enqueue_kernel for MPI
}

/**
 * @brief For the kernel given by idx generate the set of kernel indices that
 * must be executed before kernel idx.
 *
 * @param idx the index of the kernel
 *
 * @return set of indices of kernels the kenerl_idx depends on.
 */
std::set<int> get_dependencies(unsigned idx) {
  std::set<int> dependencies;
  ops_kernel_descriptor *desc = ops_kernel_dag[idx];
  // check loop dependencies
  for (int param_idx = 0; param_idx < desc->nargs; param_idx++) {
    ops_arg arg = desc->args[param_idx];
    if (arg.argtype == OPS_ARG_GBL) {
      continue;
    } else if (arg.argtype == OPS_ARG_DAT) {
      ops_dat dat = arg.dat;
      std::string name = dat->name;

      if ((arg.acc == OPS_READ && 0 != ops_dat_states.count(name)) ||
          arg.acc == OPS_INC || arg.acc == OPS_RW) {
        // get an iterator to the first greater or equal element to the idx of
        // the kernel
        auto it = std::lower_bound(ops_dat_states[name].begin(),
                                   ops_dat_states[name].end(), idx);
        if (it != ops_dat_states[name]
                      .begin()) { // TODO make sure we dont need the if.. ie
                                  // store the initial version properly
          it--; // but we need the last smaller so decrease the iterator
          dependencies.insert(*it);
        } else {
          ops_printf("para\n");
        }
      }
    }
  }
  return dependencies;
}

void ops_run_kernel(unsigned idx) {
  // we will use flood fill or some variant to get the set of all kernels that I
  // want to execute..
  std::set<int> loops;
  std::set<int> visited;
  std::set<int> to_be_visit = get_dependencies(idx);
  while (!to_be_visit.empty()) {
    int cur = *to_be_visit.begin();
    to_be_visit.erase(to_be_visit.begin());
    if (visited.count(cur)) {
      continue;
    }
    loops.insert(cur);
    visited.insert(cur);
    std::set<int> children = get_dependencies(cur);
    to_be_visit.insert(children.begin(), children.end());
  }
  // TODO reset memory..
  for (const int &dep_idx : loops) {
    ops_printf("rerun kernel %u: %s\n", dep_idx, ops_kernel_dag[dep_idx]->name);
    ops_kernel_dag[dep_idx]->function(ops_kernel_dag[dep_idx]);
  }
  ops_printf("rerun kernel %u: %s\n", idx, ops_kernel_dag[idx]->name);
  ops_kernel_dag[idx]->function(ops_kernel_dag[idx]);
}

void ops_traverse_dag() {
  // TODO reset memory!
  for (unsigned i = 0; i < ops_kernel_dag.size(); ++i) {
    ops_printf("rerun kernel %u: %s\n", i, ops_kernel_dag[i]->name);
    ops_kernel_dag[i]->function(ops_kernel_dag[i]);
  }
}

