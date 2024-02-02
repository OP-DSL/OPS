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
 * @brief OPS AD library functions
 * @author Gabor Daniel Balogh
 * @details Implementations of the AD library functions utilized by all OPS
 * backends
 */
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "../algodiff/ops_dag_nodes.hpp"
#include "ops_algodiff.hpp"
#include "ops_lib_core.h"

const size_t ReusableMonotonicAllocator::minChunkSize; // = 1 << 20;
const size_t ReusableMonotonicAllocator::maxChunkSize; // = 1 << 30;
////////////////////////////////////////////////////////////////////////////////
//  ops_ad_dat_state - following the state of datasets, and retrieve cp-s
////////////////////////////////////////////////////////////////////////////////
ops_ad_dat_state::ops_ad_dat_state(ops_dat dat,
                                   ReusableMonotonicAllocator *allocator)
    : allocator(allocator), dat(dat) {}

char *ops_ad_dat_state::get_cp(int tape_idx) {
  assert(dat_states.size() > 0);
  if (static_cast<size_t>(cp_idx) == dat_states.size()) {
    if (dat_states[cp_idx - 1].first <= tape_idx) {
      return dat_states[cp_idx - 1].second;
    }
    cp_idx--;
  }
  if (dat_states[cp_idx].first > tape_idx) {
    for (; cp_idx > 0 && dat_states[cp_idx].first > tape_idx; cp_idx--) {
      if (dat_states[cp_idx - 1].first <= tape_idx) {
        return dat_states[cp_idx - 1].second;
      }
    }
    assert(false && "NO viable checkpoint for ops_dat");
  }
  for (; static_cast<size_t>(cp_idx) < dat_states.size() &&
         dat_states[cp_idx].first <= tape_idx;
       cp_idx++) {
  }
  assert(dat_states[cp_idx - 1].first <= tape_idx &&
         "NO viable checkpoint for ops_dat");
  return dat_states[cp_idx - 1].second;
}

int ops_ad_dat_state::num_checkpoints() const { return dat_states.size(); }

void ops_ad_dat_state::clear() {
  cp_idx = 0;
  for (auto &cp : dat_states) {
    allocator->free(cp.second);
  }
  dat_states.clear();
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

void ops_copy_host(char *dst, const char *src, int bytes) {
  for (int i = 0; i < bytes; ++i) {
    dst[i] = src[i];
  }
}

ReusableMonotonicAllocator *ops_ad_tape::get_permanent_allocator() {
  return &external_function_allocator;
}

void ops_ad_tape::resize_used_args(size_t dat_num, size_t scalar_num,
                                   size_t reduction_num) {
  if (dat_num > dats_used.size()) {
    dats_used.resize(dat_num);
  }
  if (scalar_num > scalars_used.size()) {
    scalars_used.resize(scalar_num);
  }
  if (reduction_num > reductions_used.size()) {
    reductions_used.resize(reduction_num);
  }
}

void ops_ad_tape::register_args(OPS_instance *instance, const ops_arg *args,
                                int nargs) {
  resize_used_args(instance->OPS_dat_index,
                   instance->ad_instance->OPS_scalar_list.size(),
                   instance->OPS_reduction_index);
  for (int i = 0; i < nargs; ++i) {
    if (args[i].argtype == OPS_ARG_DAT) {
      if (args[i].dat->index >= 0 && !dats_used[args[i].dat->index]) {
        dats_used[args[i].dat->index] =
            std::make_unique<ops_ad_dat_state>(args[i].dat, &allocator);
      }
    } else if (args[i].argtype == OPS_ARG_SCL) {
      ops_scalar scl = reinterpret_cast<ops_scalar>(args[i].data);
      scalars_used[scl->index] = scl;
    } else if (args[i].argtype == OPS_ARG_GBL && args[i].acc != OPS_READ) {
      ops_reduction handler = reinterpret_cast<ops_reduction>(args[i].data);
      reductions_used[handler->index] = handler;
    }
  }
}

size_t ops_ad_tape::register_decl_const(const char *name, int dim,
                                        const char *type, int type_size,
                                        char *data) {
  auto it = declared_consts.find(name);
  if (it == declared_consts.end()) {
    declared_consts[name].host_ptr = data;
    declared_consts[name].type = type;
    declared_consts[name].type_size = type_size;
    declared_consts[name].dim = dim;
    declared_consts[name].states.push_back(
        std::vector<char>(data, data + dim * type_size));
    return 0;
  }

  declared_consts[name].states.push_back(
      std::vector<char>(data, data + dim * type_size));
  return declared_consts[name].states.size() - 1;
}

void ops_ad_tape::reset_const_to(OPS_instance *instance,
                                 const std::string &name, size_t pos) {
  auto it = declared_consts.find(name);
  assert(it != declared_consts.end());
  assert(pos < it->second.states.size());
  ops_decl_const_char(instance, it->second.dim, it->second.type.c_str(),
                      it->second.type_size, it->second.states[pos].data(),
                      name.c_str());
  ops_copy_host(it->second.host_ptr, it->second.states[pos].data(),
                it->second.type_size * it->second.dim);
}

void ops_ad_tape::zero_adjoints_except(const std::vector<ops_dat> &exceptions) {
  for (auto &record : dats_used) {
    if (!record) {
      continue;
    }
    ops_dat dat = record->dat;
    if (dat && !dat->is_passive &&
        exceptions.end() !=
            std::find(exceptions.begin(), exceptions.end(), dat)) {
      ops_init_zero(dat->derivative, dat->mem);
      if (dat->derivative_d != nullptr) {
        // passing a nullptr as host memory will call cudaMemset with 0
        ops_cpHostToDevice(dat->block->instance, (void **)&(dat->derivative_d),
                           nullptr, dat->mem);
      }
      dat->ad_dirty_hd = 0;
    }
  }
  for (size_t i = 0; i < scalars_used.size(); ++i) {
    if (scalars_used[i] && scalars_used[i]->derivative) {
      ops_init_zero(scalars_used[i]->derivative, scalars_used[i]->size);
    }
  }
  for (ops_reduction handler : reductions_used) {
    if (handler && handler->derivative) {
      ops_init_zero(handler->derivative, handler->size);
    }
  }
}

void ops_ad_tape::zero_adjoints() { zero_adjoints_except({}); }

void ops_ad_tape::checkpoint_all() {
  if (final_state_dats.size() == dats_used.size()) {
    // Already created checkpoints, nothing to do
    return;
  }
  final_state_dats.resize(dats_used.size(), nullptr);
  final_state_scalars.resize(scalars_used.size());
  final_state_reductions.resize(reductions_used.size());
  for (size_t i = 0; i < dats_used.size(); ++i) {
    if (dats_used[i] && dats_used[i]->num_checkpoints() > 0) {
      if (nullptr == final_state_dats[i]) {
        final_state_dats[i] =
            external_function_allocator.allocate(dats_used[i]->dat->mem);
      }
      ops_copy_cp_to_buf(dats_used[i]->dat, final_state_dats[i]);
      // This optimisation sounds nice, but args hold pointers to the original
      // data... if (nullptr == dats_used[i]->dat->data_d) {
      //   std::swap(dats_used[i]->dat->data, final_state_dats[i]);
      // } else {
      //   std::swap(dats_used[i]->dat->data_d, final_state_dats[i]);
      // }
      dats_used[i]->dat->dirty_hd =
          dats_used[i]->dat->data_d ? OPS_DEVICE : OPS_HOST;
    }
  }
  for (size_t i = 0; i < scalars_used.size(); ++i) {
    if (scalars_used[i]) { // TODO save only changed scalars
      ops_scalar scl = scalars_used[i];
      final_state_scalars[i].resize(scl->size);
      std::copy(scl->data, scl->data + scl->size,
                final_state_scalars[i].begin());
    }
  }
  for (size_t i = 0; i < reductions_used.size(); ++i) {
    if (reductions_used[i]) {
      ops_reduction handler = reductions_used[i];
      final_state_reductions[i].resize(handler->size);
      std::copy(handler->data, handler->data + handler->size,
                final_state_reductions[i].begin());
    }
  }
}

void ops_ad_tape::load_final_state(OPS_instance *instance) {
  for (size_t i = 0; i < dats_used.size(); ++i) {
    if (dats_used[i] && final_state_dats[i]) {
      // This optimisation sounds nice, but args hold pointers to the original
      // data... if (dats_used[i]->dat->data_d) {
      //   std::swap(final_state_dats[i], dats_used[i]->dat->data_d);
      // } else {
      //   std::swap(final_state_dats[i], dats_used[i]->dat->data);
      // }
      ops_copy_cp(dats_used[i]->dat, final_state_dats[i]);
      dats_used[i]->dat->dirty_hd =
          dats_used[i]->dat->data_d ? OPS_DEVICE : OPS_HOST;
    }
  }
  for (size_t i = 0; i < scalars_used.size(); ++i) {
    if (scalars_used[i] && final_state_scalars[i].size() > 0) {
      ops_scalar scl = scalars_used[i];
      std::copy(final_state_scalars[i].begin(), final_state_scalars[i].end(),
                scl->data);
    }
  }
  for (size_t i = 0; i < reductions_used.size(); ++i) {
    if (reductions_used[i]) {
      ops_reduction handler = reductions_used[i];
      std::copy(final_state_reductions[i].begin(),
                final_state_reductions[i].end(), handler->data);
    }
  }
  for (auto &it : declared_consts) {
    if (it.second.states.size() > 1) {
      ops_decl_const_char(instance, it.second.dim, it.second.type.c_str(),
                          it.second.type_size, it.second.states.back().data(),
                          it.first.c_str());
      ops_copy_host(it.second.host_ptr, it.second.states.back().data(),
                    it.second.type_size * it.second.dim);
    }
  }
}

void ops_ad_set_checkpoint_count(int max_available_cp, int max_chain_length,
                                 int max_recompute_num) {
  ops_ad_set_checkpoint_count(OPS_instance::getOPSInstance(), max_available_cp,
                              max_chain_length, max_recompute_num);
}

void ops_ad_set_checkpoint_count(OPS_instance *instance, int max_available_cp,
                                 int max_chain_length, int max_recompute_num) {
  assert(instance->ad_instance);
  instance->ad_instance->get_default_tape()->init_revolve(
      max_available_cp, max_chain_length, max_recompute_num);
}

void ops_ad_tape::init_revolve(int max_available_cp, int max_chain_length,
                               int max_recompute_num) {
  assert(dagState == OPSDagState::FORWARD_SAVE);
  assert(strategy == ops_ad_tape::checkpointing_strategy::SAVE_ALL);
  strategy = ops_ad_tape::checkpointing_strategy::REVOLVE;
  revolve.max_cp_count = max_available_cp;
  revolve.length_of_chain = max_chain_length;
  if (max_recompute_num == -1) {
    revolve.compute_recompute_num();
  } else {
    revolve.max_recompute_num = max_recompute_num;
  }
  dagState = OPSDagState::FORWARD;
}

// compute the maximum number of recomputes per node given a chain length and
// checkpoint count
void ops_ad_tape::revolve_handler::compute_recompute_num() {
  max_recompute_num = 1;
  double t_factor = 1;
  double st = (max_cp_count + max_recompute_num);
  double beta = st / t_factor;
  while (length_of_chain > beta) {
    max_recompute_num++;
    st *= max_cp_count + max_recompute_num;
    t_factor *= max_recompute_num;
    beta = st / t_factor;
  }
}

// compute \beta(s,t) = (s+t)!/(s!t!)
int ops_ad_tape::revolve_handler::beta(int s, int t) {
  if (t < 0 || s < 0) {
    return 0;
  }
  // compute the binomial
  double beta = std::exp(std::lgamma(s + t + 1) - std::lgamma(t + 1) -
                         std::lgamma(s + 1));
  return std::round(beta);
}

int ops_ad_tape::revolve_handler::m_hat(int m, int s, int t) {
  if (m <= beta(s, t - 1) + beta(s - 2, t - 1))
    return beta(s, t - 2);
  if (m >= beta(s, t) - beta(s - 3, t))
    return beta(s, t - 1);
  return m - beta(s - 1, t - 1) - beta(s - 2, t - 1);
}

void ops_ad_tape::add_to_dag(std::unique_ptr<ops_dag_node> &&node) {
  current_tape_idx = dag.size();
  dag.push_back(std::move(node));
}

ops_scalar_core::ops_scalar_core(OPS_instance *instance, const char *_data,
                                 std::string _name, int dim, int t_size,
                                 int index) noexcept
    : name(_name), data((char *)ops_malloc(dim * t_size)),
      derivative((char *)ops_malloc(dim * t_size)), data_d(nullptr),
      derivative_d(nullptr), dirty_hd(0), ad_dirty_hd(0), dim(dim),
      size(dim * t_size), index(index), instance(instance) {
  memcpy(data, _data, size);
  ops_cpHostToDevice(instance, (void **)&data_d, (void **)&data, size);
  ops_init_zero(derivative, size);
  ops_cpHostToDevice(instance, (void **)&derivative_d, nullptr, size);
}

ops_scalar_core::~ops_scalar_core() {
  if (data)
    ops_free(data);
  if (derivative)
    ops_free(derivative);
  ops_free_scalar_device(this);
}

class ops_dat_derivative_wrapper {
  ops_dat derivative_dat;

public:
  const int idx;
  ops_dat get_derivative() const { return derivative_dat; }
  ops_dat_derivative_wrapper(ops_dat primal) noexcept : idx(primal->index) {
    assert(primal->block->instance->ad_instance);
    derivative_dat = (ops_dat)ops_malloc(sizeof(ops_dat_core));
    *derivative_dat = *primal;
    derivative_dat->data = primal->derivative;
    derivative_dat->data_d = primal->derivative_d;
    derivative_dat->derivative = nullptr;
    derivative_dat->derivative_d = nullptr;
    derivative_dat->dirty_hd = primal->ad_dirty_hd;
    derivative_dat->ad_dirty_hd = 0;
    derivative_dat->locked_hd = 0;
    derivative_dat->user_managed = 1;
    derivative_dat->is_passive = true;
    // derivative_dat->name = ???
    derivative_dat->index = -1 - primal->index;
  }
  ops_dat_derivative_wrapper(const ops_dat_derivative_wrapper &) = delete;
  ops_dat_derivative_wrapper(ops_dat_derivative_wrapper &&) = delete;
  ~ops_dat_derivative_wrapper() {
    ops_free(derivative_dat);
    derivative_dat = nullptr;
  }
};
/////////////////////////////////////////////////////////////////////////
// State search for datasets and other utility functions
/////////////////////////////////////////////////////////////////////////

char *ops_get_cp(ops_dat dat) {
  OPS_instance *instance = dat->block->instance;
  return instance->ad_instance->get_current_tape()->get_current_checkpoint_for(
      dat);
}

char *ops_ad_tape::get_current_checkpoint_for(ops_dat dat) {
  assert(dats_used[dat->index] && "CP access on new dat");
  return dats_used[dat->index]->get_cp(interpretedNode);
}

/////////////////////////////////////////////////////////////////////////
// Checkpoint allocation and restoration for ops_dat
// Allocation of adjoints
/////////////////////////////////////////////////////////////////////////

void ops_copy_scalar_cp(char *dst, const char *src, int bytes) {
  memcpy(dst, src, bytes);
}

void OPS_instance_ad::allocate_ad(ops_dat dat) {
  if (dat->is_passive)
    return;
  if (dat->derivative == nullptr) {
    dat->derivative = (char *)ops_malloc(dat->mem);
    ops_init_zero(dat->derivative, dat->mem);
  }
  if (dat->derivative_d == nullptr) {
    // passing a nullptr as host memory will call cudaMemset with 0
    ops_cpHostToDevice(dat->block->instance, (void **)&(dat->derivative_d),
                       nullptr, dat->mem);
  }
}

void OPS_instance_ad::allocate_ad(ops_reduction handle) {
  if (handle->derivative == nullptr) {
    handle->derivative = static_cast<char *>(ops_malloc(handle->size));
    ops_init_zero(handle->derivative, handle->size);
  }
}

OPS_instance_ad::OPS_instance_ad() noexcept {
  tapes.emplace_back(std::make_unique<ops_ad_tape>());
}

ops_ad_tape *OPS_instance_ad::create_tape() {
  if (get_current_tape()->dagState != OPSDagState::INTERPRET_EXTERNAL) {
    OPSException exc(OPS_INTERNAL_ERROR);
    exc << "ops_create_tape must be called from the adjoint fo an external "
           "function";
    throw exc;
  }
  tapes.emplace_back(std::make_unique<ops_ad_tape>());
  return get_current_tape();
}

void OPS_instance_ad::remove_tape(ops_ad_tape *tape) {
  if (tapes.size() <= 1) {
    OPSException exc(OPS_INTERNAL_ERROR);
    exc << "ops_remove_tape: cannot remove global tape";
    throw exc;
  }
  if (get_current_tape() != tape) {
    OPSException exc(OPS_INTERNAL_ERROR);
    exc << "ops_remove_tape: only the last tape can be destroyed";
    throw exc;
  }
  tapes.pop_back();
}

ops_ad_tape *OPS_instance_ad::get_current_tape() { return tapes.back().get(); }

ops_ad_tape *OPS_instance_ad::get_default_tape() { return tapes[0].get(); }

ops_dat OPS_instance_ad::get_derivative_as_dat(ops_dat dat) {
  for (const auto &wrapper : derivative_dats) {
    if (wrapper->idx == dat->index) {
      return wrapper->get_derivative();
    }
  }
  derivative_dats.push_back(std::make_unique<ops_dat_derivative_wrapper>(dat));
  return derivative_dats.back()->get_derivative();
}

ops_ad_checkpoint::ops_ad_checkpoint(const std::vector<ops_dat> &datlist,
                                     CheckpointAllocator &allocator,
                                     bool do_save)
    : allocator(&allocator) {
  for (auto &dat : datlist) {
    checkpoints.emplace_back(dat, nullptr);
  }
  if (do_save) {
    save();
  }
}

void ops_ad_checkpoint::free() {
  for (auto &cp : checkpoints) {
    cp.second = nullptr;
  }
  if (cp_start) {
    allocator->free(cp_start);
    cp_start = nullptr;
  }
}

bool ops_ad_checkpoint::is_active() const { return cp_start; }
int ops_ad_checkpoint::reruns() const { return run_count; }
void ops_ad_checkpoint::increment_run_count() { run_count++; }

void ops_ad_checkpoint::save() {
  assert(cp_start == nullptr);
  int cp_size =
      std::accumulate(checkpoints.begin(), checkpoints.end(), 0,
                      [](const int &sum, const std::pair<ops_dat, char *> &cp) {
                        return sum + cp.first->mem;
                      });
  char *cp = allocator->allocate(cp_size);
  cp_start = cp;
  for (std::pair<ops_dat, char *> &cp_entry : checkpoints) {
    ops_copy_cp_to_buf(cp_entry.first, cp);
    cp_entry.second = cp;
    cp += cp_entry.first->mem;
  }
}

void ops_ad_checkpoint::load() const {
  for (const auto &cp : checkpoints) {
    ops_copy_cp(cp.first, cp.second);
  }
}

ops_ad_checkpoint::ops_ad_checkpoint(ops_ad_checkpoint &&other)
    : checkpoints(other.checkpoints), cp_start(other.cp_start),
      allocator(other.allocator), run_count(other.run_count) {
  other.cp_start = nullptr;
  other.checkpoints.clear();
}

ops_ad_checkpoint::~ops_ad_checkpoint() { free(); }

void ops_ad_manual_checkpoint_datlist_impl(
    const std::vector<ops_dat> &datlist) {
  OPS_instance *instance = datlist.at(0)->block->instance;
  assert(instance->ad_instance->get_default_tape() ==
         instance->ad_instance->get_current_tape());
  instance->ad_instance->get_default_tape()->create_checkpoint(datlist);
}

void ops_ad_tape::create_checkpoint(const std::vector<ops_dat> &datlist) {
  assert(strategy == ops_ad_tape::checkpointing_strategy::REVOLVE);
  assert(revolve.checkpoints.size() <
         static_cast<size_t>(revolve.length_of_chain));
  bool do_save = false;
  if (revolve.checkpoints.size() == static_cast<size_t>(revolve.next_cp_idx) &&
      revolve.cp_count < revolve.max_cp_count) {
    do_save = true;
    revolve.next_cp_idx += revolve_handler::m_hat(
        revolve.length_of_chain - revolve.checkpoints.size(),
        revolve.max_cp_count - revolve.cp_count, revolve.max_recompute_num);
    revolve.cp_count++;
  }
  revolve.checkpoints.emplace_back(
      dag.size(), ops_ad_checkpoint(datlist, revolve.allocator, do_save));
  if (revolve.checkpoints.size() ==
      static_cast<size_t>(revolve.length_of_chain)) {
    dagState = OPSDagState::FORWARD_SAVE;
  }
}

ops_dag_node *ops_ad_tape::get_active_node() {
  return dag[current_tape_idx].get();
}

/////////////////////////////////////////////////////////////////////////
//  Check access in external nodes
/////////////////////////////////////////////////////////////////////////

#define tape instance->ad_instance->get_current_tape()
#define dag instance->ad_instance->get_current_tape()->dag
#define dagState instance->ad_instance->get_current_tape()->dagState
void ops_ad_check_access(ops_dat dat) {
  OPS_instance *instance = dat->block->instance;
  if (dat->index < 0) {
    OPSException exp(OPS_INVALID_ARGUMENT);
    exp << "ERROR: access to raw pointers or fetch data from an ops_dat "
           "representing a derivative";
    throw exp;
  }
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())->check_access(dat);
  }
}

void ops_ad_check_release(ops_dat dat, ops_access acc) {
  OPS_instance *instance = dat->block->instance;
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())
        ->check_release(dat, acc);
  }
}

void ops_ad_check_access(ops_scalar scalar) {
  OPS_instance *instance = scalar->instance;
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())
        ->check_access(scalar);
  }
}

void ops_ad_check_release(ops_scalar scalar, ops_access acc) {
  OPS_instance *instance = scalar->instance;
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())
        ->check_release(scalar, acc);
  }
}

void ops_ad_create_dat(ops_dat dat) {
  OPS_instance *instance = dat->block->instance;
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())->create_dat(dat);
  }
}
void ops_ad_create_scalar(ops_scalar scl) {
  OPS_instance *instance = scl->instance;
  if (instance->ad_instance && dagState == OPSDagState::EXTERNAL) {
    ((ops_external_function_node *)tape->get_active_node())->create_scalar(scl);
  }
}

void ops_ad_check_access_derivative(ops_dat dat) {
  if (dat->index < 0) {
    OPSException exp(OPS_INVALID_ARGUMENT);
    exp << "Request to derivative for an ops_dat representing a derivative "
           "op "
        << dat->name;
    throw exp;
  }
  if (dat->is_passive) {
    std::string err = "Request derivative of a passive ops_dat(";
    err += dat->name;
    err += ")";
    throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
  }
  OPS_instance *instance = dat->block->instance;
  if (instance->ad_instance && dagState == OPSDagState::INTERPRET_EXTERNAL) {
    ((ops_external_function_node *)dag
         .at(instance->ad_instance->get_current_tape()->interpretedNode)
         .get())
        ->check_access_derivative(dat);
  } else if (dat->derivative == nullptr) {
    instance->ad_instance->allocate_ad(dat);
  }
}

void ops_ad_check_release_derivative(ops_dat dat) {
  OPS_instance *instance = dat->block->instance;
  if (instance->ad_instance && dagState == OPSDagState::INTERPRET_EXTERNAL) {
    ((ops_external_function_node *)dag
         .at(instance->ad_instance->get_current_tape()->interpretedNode)
         .get())
        ->check_release_derivative(dat);
  }
}

ops_dat ops_get_derivative_as_ops_dat(ops_dat dat) {
  OPS_instance *instance = dat->block->instance;
  if (!instance->ad_instance)
    return nullptr;
  if (dat->index < 0) {
    OPSException exp(OPS_INVALID_ARGUMENT);
    exp << "ERROR: Accessing derivative of an ops_dat representing the "
           "derivative of an other ops_dat";
    throw exp;
  }
  return instance->ad_instance->get_derivative_as_dat(dat);
}
#undef dag
#undef dagState
#undef tape

/////////////////////////////////////////////////////////////////////////
// Scalar related functions
// - ops_arg declare for scalar arguments
// - ops_decl_scalar_core
/////////////////////////////////////////////////////////////////////////

ops_arg ops_arg_scalar(ops_scalar data, int dim, char const *type,
                       ops_access acc) {
  ops_arg arg =
      ops_arg_scalar_core((char *)data, dim, data->size / data->dim, acc);
  arg.data_d = data->data_d;
  arg.derivative = data->derivative;
  arg.derivative_d = data->derivative_d;
  return arg;
}

ops_scalar ops_decl_scalar_core(OPS_instance *instance, std::string name,
                                const char *data, int dim, int t_size) {
  assert(instance->ops_enable_ad && "scalars only implemented for AD");
  if (instance->ops_enable_ad && instance->ad_instance == nullptr)
    instance->ad_instance = new OPS_instance_ad();
  instance->ad_instance->OPS_scalar_list.emplace_back(
      new ops_scalar_core(instance, data, name, dim, t_size,
                          instance->ad_instance->OPS_scalar_list.size()));

  return instance->ad_instance->OPS_scalar_list.back().get();
}

/////////////////////////////////////////////////////////////////////////
// Backward run
/////////////////////////////////////////////////////////////////////////

void allocate_derivatives(OPS_instance *instance) {
  if (!instance->ad_instance)
    return;
  /* allocate derivative arrays */
  ops_dat_entry *item = TAILQ_FIRST(&instance->OPS_dat_list);
  while (item) {
    instance->ad_instance->allocate_ad(item->dat);
    item = TAILQ_NEXT(item, entries);
  }
}

void ops_ad_tape::interpret_section(OPS_instance *instance, int dag_rlast_idx) {
  dagState = OPSDagState::INTERPRET;
#ifndef NDEBUG
  instance->ostream() << "bw: " << interpretedNode << "->" << dag_rlast_idx
                      << "\n";
#endif
  for (; interpretedNode >= dag_rlast_idx; --interpretedNode) {
    // #ifndef NDEBUG
    //     instance->ostream() << dag[interpretedNode]->getName() << "\n";
    // #endif
    dag[interpretedNode]->interpret_adjoint();
  }
}

void ops_ad_tape::advance_section(OPS_instance *instance, int dag_begin_idx,
                                  int dag_end_idx) {
#ifndef NDEBUG
  instance->ostream() << "re: " << dag_begin_idx << "->" << dag_end_idx << "\n";
#endif
  for (current_tape_idx = dag_begin_idx; current_tape_idx < dag_end_idx;
       ++current_tape_idx) {
    // #ifndef NDEBUG
    //     instance->ostream() << "re: " << dag[current_tape_idx]->getName() <<
    //     "\n";
    // #endif
    dag[current_tape_idx]->visit_node();
  }
}

void ops_ad_tape::interpret_adjoint(OPS_instance *instance) {
  if (!instance->ad_instance) {
    OPSException exp(OPS_INTERNAL_ERROR);
    exp << "ERROR interpret_adjoint with uninitialized adjoint instance";
    throw exp;
  }
  if (interpretedNode == -1) {
    allocate_derivatives(instance); // TODO rework
    interpretedNode = dag.size() - 1;
  } else {
    OPSException exp(OPS_INVALID_ARGUMENT);
    exp << "ERROR interpret_adjoint called multiple times before finish on "
           "the same tape";
    throw exp;
  }
  // store final state to restore at the end
  checkpoint_all();
  try {
    if (strategy == checkpointing_strategy::SAVE_ALL) {
      interpret_section(instance, 0);
    } else { // REVOLVE
      if (revolve.checkpoints.size() !=
          static_cast<size_t>(revolve.length_of_chain)) {
        int old_len = revolve.length_of_chain;
        revolve.length_of_chain = revolve.checkpoints.size();
        revolve.compute_recompute_num();
        if (revolve.checkpoints.size() > static_cast<size_t>(old_len)) {
          revolve.max_recompute_num++;
        }
      }
#ifndef NDEBUG
      ops_printf2(instance, "revolve: chain size: %lu, active cp: %d\n",
                  revolve.checkpoints.size(), revolve.cp_count);
#endif

      for (int icp_idx = revolve.checkpoints.size() - 1; icp_idx >= 0;
           --icp_idx) {
#ifndef NDEBUG
        ops_printf2(instance, "revolve: cp_idx: %d, active cp: %d\n", icp_idx,
                    revolve.cp_count);
#endif
        // if the last section was not checkpointed we need to rerun and do cp
        // it
        if (dagState != OPSDagState::FORWARD_SAVE) {
          // step 1: search last taken checkpoint
          int rerun_idx = icp_idx;
          while (!revolve.checkpoints[rerun_idx].second.is_active()) {
            rerun_idx--;
          }
#ifndef NDEBUG
          ops_printf2(instance, "revolve: cp_idx: %d, rerun_from: %d\n",
                      icp_idx, rerun_idx);
#endif
          // step 2: load it
          revolve.checkpoints[rerun_idx].second.load();
          // step 3: until we reach our cp and perform forwards, calculating
          // new cp positions
          if (rerun_idx != icp_idx) {
            int next_cp_idx =
                rerun_idx +
                revolve_handler::m_hat(
                    icp_idx - rerun_idx + 1,
                    revolve.max_cp_count - revolve.cp_count + 1,
                    revolve.max_recompute_num -
                        revolve.checkpoints[rerun_idx].second.reruns());
            if (rerun_idx < icp_idx) {
              dagState = OPSDagState::FORWARD;
              advance_section(instance, revolve.checkpoints[rerun_idx].first,
                              revolve.checkpoints[rerun_idx + 1].first);
              revolve.checkpoints[rerun_idx].second.increment_run_count();
              rerun_idx++;
            }
            for (; rerun_idx < icp_idx; rerun_idx++) {
              if (next_cp_idx == rerun_idx &&
                  revolve.max_cp_count > revolve.cp_count) {
#ifndef NDEBUG
                ops_printf2(instance, "revolve: cp_idx: %d, save: %d\n",
                            icp_idx, next_cp_idx);
#endif
                assert(revolve.cp_count < revolve.max_cp_count);
                revolve.checkpoints[rerun_idx].second.save();
                revolve.cp_count++;
                next_cp_idx += revolve_handler::m_hat(
                    icp_idx - rerun_idx + 1,
                    revolve.max_cp_count - revolve.cp_count + 1,
                    revolve.max_recompute_num -
                        revolve.checkpoints[rerun_idx].second.reruns());
              }
              dagState = OPSDagState::FORWARD;
              advance_section(instance, revolve.checkpoints[rerun_idx].first,
                              revolve.checkpoints[rerun_idx + 1].first);
              revolve.checkpoints[rerun_idx].second.increment_run_count();
            }
          }
          // step 4: rerun section from icp_idx and save
          dagState = OPSDagState::FORWARD_SAVE;
          advance_section(
              instance, revolve.checkpoints[icp_idx].first,
              ((icp_idx == static_cast<int>(revolve.checkpoints.size()) - 1)
                   ? static_cast<int>(dag.size())
                   : revolve.checkpoints[icp_idx + 1].first));
        }
        // step 5: interpret current section
        interpret_section(instance, revolve.checkpoints[icp_idx].first);
        // step 6: deallocate patch type checkpoints and free revolve CP
        for (auto &dat_states : dats_used) {
          if (dat_states) {
            dat_states->clear();
          }
        }
        allocator.free_all();
        if (revolve.checkpoints[icp_idx].second.is_active()) {
          revolve.checkpoints[icp_idx].second.free();
          revolve.cp_count--;
        }
      }
    }
  } catch (...) {
    load_final_state(instance);
    throw;
  }
  load_final_state(instance);
}

void ops_interpret_adjoints(OPS_instance *instance) {
  if (!instance->ad_instance)
    return;
  instance->ad_instance->get_current_tape()->interpret_adjoint(instance);
}

void ops_interpret_adjoints() {
  ops_interpret_adjoints(OPS_instance::getOPSInstance());
}

/////////////////////////////////////////////////////////////////////////
// External functions
/////////////////////////////////////////////////////////////////////////

ops_memspace ops_ad_external_function_get_memspace(OPS_instance *instance) {
  if (instance->ad_instance &&
      instance->ad_instance->get_current_tape()->dagState ==
          OPSDagState::EXTERNAL) {
    return ((ops_external_function_node *)instance->ad_instance
                ->get_current_tape()
                ->get_active_node())
        ->get_memspace();
  }
  if (instance->ad_instance &&
      instance->ad_instance->get_current_tape()->dagState ==
          OPSDagState::INTERPRET_EXTERNAL) {
    return ((ops_external_function_node *)instance->ad_instance
                ->get_current_tape()
                ->dag
                .at(instance->ad_instance->get_current_tape()->interpretedNode)
                .get())
        ->get_memspace();
  }
  OPSException ex(OPS_RUNTIME_ERROR);
  ex << "ops_ad_external_function_get_memspace must be called from an "
        "external "
        "function";
  throw ex;
  return OPS_HOST;
}

char *ops_dat_get_raw_ptr_as_input(ops_dat dat, int part,
                                   ops_memspace *memspace) {
  OPS_instance *instance = dat->block->instance;
  if (instance->ad_instance) {
    if (instance->ad_instance->get_current_tape()->dagState !=
        OPSDagState::INTERPRET_EXTERNAL) {
      OPSException ex(OPS_RUNTIME_ERROR);
      ex << "ops_dat_get_raw_ptr_as_input called outside of adjoints of "
            "external functions";
      throw ex;
    }
    reinterpret_cast<ops_external_function_node *>(
        instance->ad_instance->get_current_tape()
            ->dag[instance->ad_instance->get_current_tape()->interpretedNode]
            .get())
        ->load_state_before(dat);
  }
  return ops_dat_get_raw_pointer(dat, part, nullptr, memspace);
}

ops_ad_tape *ops_create_tape(OPS_instance *instance) {
  if (instance->ad_instance == nullptr) {
    OPSException ex(OPS_RUNTIME_ERROR);
    throw ex;
  }
  return instance->ad_instance->create_tape();
}

void ops_remove_tape(OPS_instance *instance, ops_ad_tape *tape) {
  if (instance->ad_instance == nullptr) {
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "ops_remove_tape must be called form AD active code";
    throw ex;
  }
  instance->ad_instance->remove_tape(tape);
}

/////////////////////////////////////////////////////////////////////////
// derivative allocation and exit
/////////////////////////////////////////////////////////////////////////

void ops_ad_alloc_derivative(ops_reduction handle) {
  if (handle->instance->ad_instance) {
    handle->instance->ad_instance->allocate_ad(handle);
  }
}

// This funtion called from OPS_instance destructor
void ops_exit_ad(OPS_instance *instance) {
  if (instance->ad_instance == nullptr)
    return;
  delete instance->ad_instance;
  instance->ad_instance = nullptr;
}
