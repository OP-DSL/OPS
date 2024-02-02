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
 * @brief OPS AD library dag node implementations
 * @author Gabor Daniel Balogh
 * @details Implementations of the DAG nodes utilized by all OPS backends
 */
#ifndef OPS_ALGODIFF_DAG_NODE_HPP_INCLUDED
#define OPS_ALGODIFF_DAG_NODE_HPP_INCLUDED

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ops_lib_core.h"
#if __cplusplus < 201402L
// useful bits and pieces from c++14
namespace std {
template <class T, class... Args> unique_ptr<T> make_unique(Args &&...args) {
  return unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace std
#endif

void ops_copy_host(char *dst, const char *src, int bytes);

/**
 * @brief Abstract node type for building DAG inside OPS
 */
class ops_dag_node {
protected:
  int idx; /**< The position inside the DAG */
public:
  ops_dag_node(int idx) noexcept : idx(idx) {}

  /**
   * @brief Visiting the node either replay the operation or set memory to the
   * state after the event
   */
  virtual void visit_node() = 0;
  /**
   * @brief Visiting the node and calculate the adjoints corresponding to the
   * node
   */
  virtual void interpret_adjoint() = 0;

  /**
   * @brief Get some nicely printable name for the node
   */
  virtual std::string getName() const = 0;

  virtual ~ops_dag_node() noexcept = default;
};

/////////////////////////////////////////////////////////////////////////
//                                Loops
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Node type representing an ops_par_loop call
 *
 */
class ops_loop_node final : public ops_dag_node {
  ops_kernel_descriptor *desc; /**< descriptor for the kernel call */
  std::vector<std::vector<char>>
      reduction_initial_values; /**< Reduction handlers are initialised when
                                   the arg is created. This value is
                                   overwritten in loops. Store the initial
                                   value so we can recompute them. The
                                   copies are in the same order as the
                                   reductions appear in the args list. */
#define instance desc->block->instance
#define allocate_ad instance->ad_instance->allocate_ad
  /**
   * @brief initialise uninitialised adjoint pointers before running adjoint
   * loop.
   */
  void prepare_adjoints() {
    for (int i = 0; i < desc->nargs; ++i) {
      if (desc->args[i].derivative != nullptr) {
        continue;
      }
      if (desc->args[i].argtype == OPS_ARG_DAT) {
        ops_dat dat = desc->args[i].dat;
        desc->args[i].derivative = dat->derivative;
        desc->args[i].derivative_d = dat->derivative_d;
      } else if (desc->args[i].argtype == OPS_ARG_GBL) {
        if (desc->args[i].acc == OPS_MAX || desc->args[i].acc == OPS_MIN ||
            desc->args[i].acc == OPS_INC) {
          ops_reduction handle = (ops_reduction)desc->args[i].data;
          if (handle->derivative == nullptr) {
            allocate_ad(handle);
          }
          desc->args[i].derivative = handle->derivative;
        }
      }
    }
  }

  /**
   * @brief Reset reductions to the state before the loop
   */
  void reset_reductions(bool before_interpret) {
    for (int red_idx = 0, param_idx = 0;
         static_cast<size_t>(red_idx) < reduction_initial_values.size() &&
         param_idx < desc->nargs;
         param_idx++) {
      ops_arg arg = desc->args[param_idx];
      if (arg.argtype == OPS_ARG_GBL) {
        if (arg.acc == OPS_MAX || arg.acc == OPS_MIN || arg.acc == OPS_INC) {
          ops_reduction handle = (ops_reduction)arg.data;
          ops_copy_host(handle->data, reduction_initial_values[red_idx].data(),
                        handle->size);
          if (before_interpret) {
            handle->initialized = 1;
          } else {
            handle->initialized = 0;
          }
          red_idx++;
        }
      }
    }
  }

public:
  ops_loop_node(int idx, ops_kernel_descriptor *_desc) noexcept
      : ops_dag_node(idx), desc(_desc) {
    for (int param_idx = 0; param_idx < desc->nargs; param_idx++) {
      ops_arg *arg = &desc->args[param_idx];
      if (arg->argtype == OPS_ARG_GBL) {
        if (arg->acc == OPS_MAX || arg->acc == OPS_MIN || arg->acc == OPS_INC) {
          ops_reduction handle = (ops_reduction)arg->data;
          reduction_initial_values.emplace_back(handle->size);
          ops_copy_host(reduction_initial_values.back().data(), handle->data,
                        handle->size);
        }
      }
    }
  }

  ~ops_loop_node() {
    for (int i = 0; i < desc->nargs; ++i) {
      if (desc->args[i].argtype == OPS_ARG_GBL &&
          desc->args[i].acc == OPS_READ) {
        free(desc->args[i].data);
      }
    }
    ops_free(desc->args);
    desc->args = nullptr;
    ops_free(desc);
  }

  /**
   * @brief Execute the loop
   */
  void visit_node() override {
    reset_reductions(true);
    desc->function(desc);
  }

  /**
   * @brief Run the adjoint kernel
   */
  void interpret_adjoint() override {
    if (desc->adjoint_function) { // TODO @bgd fix double check on active loop
      // interpret adjoints if the loop is not passive
      if (desc->adjoint_function) {
        reset_reductions(true);
        prepare_adjoints();
        desc->adjoint_function(desc);
      }
      reset_reductions(false);
    }
  }

  std::string getName() const override { return desc->name; }
#undef allocate_ad
#undef instance
};

//==================== Add loops to DAG ====================

void ops_add_to_dag(ops_kernel_descriptor *desc) {
  OPS_instance *instance = desc->block->instance;
  if (instance->ops_enable_ad && instance->ad_instance == nullptr)
    instance->ad_instance = new OPS_instance_ad();
  if (instance->ad_instance) {
    instance->ad_instance->get_current_tape()->register_args(
        instance, desc->args, desc->nargs);
  }
#define dagState instance->ad_instance->get_current_tape()->dagState
#define dag instance->ad_instance->get_current_tape()->dag
  if (instance->ad_instance && (dagState == OPSDagState::FORWARD ||
                                dagState == OPSDagState::FORWARD_SAVE)) {
    double t1, t2, c;
    if (instance->OPS_diags > 1)
      ops_timers_core(&c, &t1);

    // Add desc to DAG
    instance->ad_instance->get_current_tape()->add_to_dag(
        std::make_unique<ops_loop_node>(dag.size(), desc));

    if (instance->OPS_diags > 1) {
      ops_timers_core(&c, &t2);
      instance->OPS_kernels[desc->index].dag_build_overhead += t2 - t1;
    }
    // Run the kernel
    if (desc->adjoint_function == nullptr && dagState != OPSDagState::FORWARD) {
      assert(dagState == OPSDagState::FORWARD_SAVE);
      dagState = OPSDagState::FORWARD;
      desc->function(desc);
      dagState = OPSDagState::FORWARD_SAVE;
    } else {
      desc->function(desc);
    }
  } else { // no AD or in EXTERNAL
    // Run the kernel
    desc->function(desc);
    for (int i = 0; i < desc->nargs; ++i) {
      if (desc->args[i].argtype == OPS_ARG_GBL &&
          desc->args[i].acc == OPS_READ) {
        free(desc->args[i].data);
      }
    }
    ops_free(desc->args);
    ops_free(desc);
  }
#undef dagState
#undef dag
}

/////////////////////////////////////////////////////////////////////////
//                              Reductions
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Node type representing an ops_reduction_result call in the DAG
 */
class ops_get_red_result_node final : public ops_dag_node {
  ops_reduction handle;
  std::vector<char> red_result; /**< restore red result just in case. Shouldn't
                                   matter unless OPS_INC with multiplication. */
  std::vector<char>
      derivative; /**< derivative of the result of the reduction should be given
                     prior to get_result or together with it. */
  ops_access acc; /**< save the type of the reduction. */
public:
  ops_get_red_result_node(int idx, ops_reduction _handle)
      : ops_dag_node(idx), handle(_handle), red_result(_handle->size),
        derivative(_handle->size, 0), acc(handle->acc) {
    ops_copy_host(red_result.data(), handle->data, handle->size);
    if (handle->derivative) {
      ops_copy_host(derivative.data(), handle->derivative, handle->size);
      ops_init_zero(handle->derivative, handle->size);
    }
  }

  void visit_node() override {
    // side effect of getting the result
    handle->initialized = 0;
  }

  void interpret_adjoint() override {
    // load memory
    handle->initialized = 1;
    ops_copy_host(handle->data, red_result.data(), handle->size);
    if (handle->derivative)
      ops_copy_host(handle->derivative, derivative.data(), handle->size);
    handle->acc = acc;
  }

  std::string getName() const override {
    std::string name = "ops_reduction_result handle: ";
    return name + handle->name;
  }
};

//==================== Add reduction node to DAG ====================

void ops_add_reduction_to_dag(ops_reduction handle) {
  OPS_instance *instance = handle->instance;
  if (!instance->ad_instance ||
      instance->ad_instance->get_current_tape()->dag.empty() ||
      instance->ad_instance->get_current_tape()->dagState >
          OPSDagState::FORWARD_SAVE)
    return;
  // create node, and the constructor handles the checkpointing
  instance->ad_instance->get_current_tape()->add_to_dag(
      std::make_unique<ops_get_red_result_node>(
          instance->ad_instance->get_current_tape()->dag.size(), handle));
}

/////////////////////////////////////////////////////////////////////////
//                            Global Consts
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Node type representing an ops_decl_const or ops_update_const call in
 * the DAG
 */
class ops_decl_const_node final : public ops_dag_node {
  OPS_instance *instance;
  ops_ad_tape *tape;
  std::string name;
  size_t const_stack_pos;

public:
  ops_decl_const_node(int idx, OPS_instance *instance, const char *name_c,
                      int dim, const char *type, int type_size, char *data)
      : ops_dag_node(idx), instance(instance),
        tape(instance->ad_instance->get_current_tape()), name(name_c),
        const_stack_pos(
            tape->register_decl_const(name_c, dim, type, type_size, data)) {}

  void visit_node() override {
    tape->reset_const_to(instance, name, const_stack_pos);
  }

  void interpret_adjoint() override {
    // load memory
    if (const_stack_pos != 0) {
      tape->reset_const_to(instance, name, const_stack_pos - 1);
    }
  }

  std::string getName() const override {
    std::string name1 = "ops_decl/update_const: ";
    return name1 + name;
  }
};

//==================== Add reduction node to DAG ====================

void ops_add_const_to_dag(OPS_instance *instance, const char *name, int dim,
                          const char *type, int type_size, char *data) {
  if (!instance->ad_instance ||
      instance->ad_instance->get_current_tape()->dagState >
          OPSDagState::FORWARD_SAVE)
    return;
  // create node, and the constructor handles the checkpointing
  instance->ad_instance->get_current_tape()->add_to_dag(
      std::make_unique<ops_decl_const_node>(
          instance->ad_instance->get_current_tape()->dag.size(), instance, name,
          dim, type, type_size, data));
}

/////////////////////////////////////////////////////////////////////////
//                           External Functions
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Represents a external adjoint function in the dag. Collect datasets
 * and scalars that are accessed in the external function to prepare and restore
 * their state.
 */
class ops_external_function_node final : public ops_dag_node {
  OPS_instance *instance;

  std::function<void()>
      primal; /**< function performing the external function */
  std::function<void()> fill_gap; /**< function performing adjoint computations
                                     for the external function */
  std::vector<ops_arg> args;
  bool is_recomputable_;

  ops_memspace
      memspace; /**< memspace in which the external adjoint is preferred (which
                   leads the least amount of data movement). */

  std::map<ops_dat, ops_access>
      accessed_data; /**< datasets that used by the external function and the
                        access pattern */
  std::map<ops_scalar, ops_access>
      accessed_scalars; /**< scalars used by the external function and the
                           access pattern */

  std::vector<ops_dat> created_dats; /**< ordered list of datasets created in
                                        the external funciton. */
  std::vector<ops_scalar> created_scalars; /**< ordered list of scalars created
                                              in the external funciton. */

  std::vector<std::pair<ops_dat, std::pair<char *, char *>>> dat_checkpoints;
  std::vector<
      std::pair<ops_scalar, std::pair<std::vector<char>, std::vector<char>>>>
      scl_checkpoints;

  struct reduction_vals {
    ops_reduction red;
    std::pair<std::vector<char>, std::vector<char>> states;
    std::pair<int, int> initialzed;
    std::pair<ops_access, ops_access> access;
  };

  std::vector<reduction_vals> red_checkpoints; /**< reductions accessed by
                                                   the external function */

  /**
   * @brief Reset written data to the state before the external function
   */
  void reset_memory() {
    for (auto &record : dat_checkpoints) {
      ops_dat dat = record.first;
      ops_copy_cp(dat, record.second.first);
      record.second.first = nullptr; // reset since it is freed after interpret
    }
    for (auto &record : scl_checkpoints) {
      ops_copy_host(record.first->data, record.second.first.data(),
                    record.first->size);
    }
    for (auto &record : red_checkpoints) {
      ops_copy_host(record.red->data, record.states.first.data(),
                    record.red->size);
      record.red->initialized = record.initialzed.first;
      record.red->acc = record.access.first;
    }
  }

  /**
   * @brief Reset written data to the state after the external function
   */
  void load_step() {
    assert(!is_recomputable_);
    for (auto &record : dat_checkpoints) {
      ops_dat dat = record.first;
      ops_copy_cp(dat, record.second.second);
    }
    for (auto &record : scl_checkpoints) {
      ops_copy_host(record.first->data, record.second.second.data(),
                    record.first->size);
    }
    for (auto &record : red_checkpoints) {
      ops_copy_host(record.red->data, record.states.second.data(),
                    record.red->size);
      record.red->initialized = record.initialzed.second;
      record.red->acc = record.access.second;
    }
    // TODO created scalars and datasets should be loaded?
  }

public:
  ops_external_function_node(int idx, std::function<void()> primal,
                             std::function<void()> fill_gap,
                             const std::vector<ops_arg> &args,
                             OPS_instance *instance,
                             ReusableMonotonicAllocator *allocator,
                             bool is_recomputable = false) noexcept
      : ops_dag_node(idx), instance(instance), primal(primal),
        fill_gap(fill_gap), args(args), is_recomputable_(is_recomputable) {
    // add up bytes sent from Host to Device and substract data on Device.
    // if final value is positive then memspace is host, and memspace is
    // device otherwise.
    int commHostToDevice = 0;
    for (const auto &arg : args) {
      if (arg.argtype == OPS_ARG_DAT) {
        accessed_data[arg.dat] = arg.acc;
        if (arg.acc != OPS_READ) {
          // TODO better approach to create and store checkpoints
          if (is_recomputable) {
            dat_checkpoints.push_back({arg.dat, {nullptr, nullptr}});
          } else {
            dat_checkpoints.push_back(
                {arg.dat, {nullptr, allocator->allocate(arg.dat->mem)}});
          }
        }
        commHostToDevice +=
            arg.dat->mem * (arg.dat->dirty_hd == OPS_HOST ? 1 : -1);
      } else if (OPS_ARG_SCL == arg.argtype) {
        ops_scalar scl = reinterpret_cast<ops_scalar>(arg.data);
        accessed_scalars[scl] = arg.acc;
        if (arg.acc != OPS_READ) {
          scl_checkpoints.push_back(
              {scl,
               {std::vector<char>(scl->size), std::vector<char>(scl->size)}});
          ops_copy_host(scl_checkpoints.back().second.first.data(), scl->data,
                        scl->size);
        }
      } else if (OPS_ARG_GBL == arg.argtype) {
        ops_reduction handle = reinterpret_cast<ops_reduction>(arg.data);
        red_checkpoints.push_back(
            {handle,
             {std::vector<char>(handle->size), std::vector<char>(handle->size)},
             {handle->initialized, 0},
             {handle->acc, 0}});
        ops_copy_host(red_checkpoints.back().states.first.data(), handle->data,
                      handle->size);
      }
    }
    this->memspace = commHostToDevice > 0 ? OPS_HOST : OPS_DEVICE;
  }

  bool is_recomputable() const { return is_recomputable_; }

  void save_after() {
    assert(!is_recomputable_);
    for (auto &checkpoint : dat_checkpoints) {
      ops_copy_cp_to_buf(checkpoint.first, checkpoint.second.second);
    }
    for (auto &checkpoint : scl_checkpoints) {
      ops_copy_host(checkpoint.second.second.data(), checkpoint.first->data,
                    checkpoint.first->size);
    }
    for (auto &checkpoint : red_checkpoints) {
      ops_copy_host(checkpoint.states.second.data(), checkpoint.red->data,
                    checkpoint.red->size);
      checkpoint.initialzed.second = checkpoint.red->initialized;
      checkpoint.access.second = checkpoint.red->acc;
    }
  }

  void cp_before() {
    if (is_recomputable() &&
        (dat_checkpoints.empty() ||
         (dat_checkpoints[0]
              .first->block->instance->ad_instance->get_current_tape()
              ->dagState != OPSDagState::FORWARD_SAVE)))
      return;
    for (auto &record : dat_checkpoints) {
      ops_dat dat = record.first;
      if (record.second.first == nullptr) {
        record.second.first = ops_alloc_cp(dat, dat->mem);
      }
      ops_copy_cp_to_buf(dat, record.second.first);
    }
  }

  void visit_node() override {
    OPSDagState old = instance->ad_instance->get_current_tape()->dagState;
    if (old == OPSDagState::FORWARD_SAVE) {
      cp_before();
    }
    if (is_recomputable_) {
      instance->ad_instance->get_current_tape()->dagState =
          OPSDagState::EXTERNAL;
      primal();
      instance->ad_instance->get_current_tape()->dagState = old;
      return;
    }
    load_step();
  };

  /**
   * @brief Run the user suplied adjoint function for the external function.
   */
  void interpret_adjoint() override {
    instance->ad_instance->get_current_tape()->dagState =
        OPSDagState::INTERPRET_EXTERNAL;
    // check if there is a tape leak
    ops_ad_tape *curr_tape = instance->ad_instance->get_current_tape();
    fill_gap();
    instance->ad_instance->get_current_tape()->dagState =
        OPSDagState::INTERPRET;
    reset_memory();
    if (instance->ad_instance->get_current_tape() != curr_tape) {
      OPSException exc(OPS_RUNTIME_ERROR);
      exc << "ERROR: one or more tape allocated with ops_create_tape hasn't "
             "been deallocated";
      throw exc;
    }
  };

  std::string getName() const override { return "EAO_" + std::to_string(idx); }

  void load_state_before(ops_dat dat) {
    for (auto &record : dat_checkpoints) {
      ops_dat _dat = record.first;
      if (_dat == dat) {
        ops_copy_cp(dat, record.second.first);
        break;
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////
  // Add external data accesses to node
  /////////////////////////////////////////////////////////////////////////

  void check_access(ops_dat dat) {
    if (accessed_data.count(dat) != 1 &&
        std::count(created_dats.begin(), created_dats.end(), dat) != 1) {
      std::string err = "Request value of an existing ops_dat(";
      err += dat->name;
      err += ") from an external function (ops_execute_external_function) ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }

  void check_release(ops_dat dat, ops_access acc) {
    int created = std::count(created_dats.begin(), created_dats.end(), dat);
    int access_count = accessed_data.count(dat);
    if (access_count == 0 && created != 1) {
      std::string err = "Set value of an ops_dat(";
      err += dat->name;
      err += ") in an ops external adjoint context ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
    if (created)
      return;
    if (acc == -1 && access_count && accessed_data[dat] == OPS_READ) {
      std::string err = "Set value of an ops_dat(";
      err += dat->name;
      err += ") marked as read only in args.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    } else if (acc != -1 && access_count && acc != accessed_data[dat]) {
      std::string err = "Relase value of an ops_dat(";
      err += dat->name;
      err += ") with different ops_access as listed in args.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }

  void check_access(ops_scalar scalar) {
    if (accessed_scalars.count(scalar) != 1 &&
        std::count(created_scalars.begin(), created_scalars.end(), scalar) !=
            1) {
      std::string err = "Request value of an existing ops_scalar(";
      err += scalar->name;
      err += ") from an external function (ops_execute_external_function) ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }
  void check_release(ops_scalar scalar, ops_access acc) {
    if (accessed_scalars.count(scalar) == 0 &&
        std::count(created_scalars.begin(), created_scalars.end(), scalar) !=
            1) {
      std::string err = "Set value of an ops_scalar(";
      err += scalar->name;
      err += ") from an external function (ops_execute_external_function) ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
    if (accessed_scalars.count(scalar) && accessed_scalars[scalar] != acc) {
      std::string err = "Release an ops_scalar(";
      err += scalar->name;
      err += ") with different ops_access as listed in args of external func.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }

  void check_access_derivative(ops_dat dat) {
    if (accessed_data.count(dat) != 1 &&
        std::count(created_dats.begin(), created_dats.end(), dat) != 1) {
      std::string err = "Request derivative of an existing ops_dat(";
      err += dat->name;
      err += ") from an external function (ops_execute_external_function) ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }

  void check_release_derivative(ops_dat dat) {
    int created = std::count(created_dats.begin(), created_dats.end(), dat);
    if (accessed_data.count(dat) == 0 && created != 1) {
      std::string err = "Set derivative of an ops_dat(";
      err += dat->name;
      err += ") in an ops external adjoint context ";
      err += "without listing it as an argument.";
      throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
    }
  }

  void create_scalar(ops_scalar scl) { created_scalars.push_back(scl); }
  void create_dat(ops_dat dat) { created_dats.push_back(dat); }
  ops_memspace get_memspace() { return memspace; }
};

//==================== Add external function to DAG ====================
void ops_execute_external_function_impl(OPS_instance *instance,
                                        std::function<void()> ext_func,
                                        std::function<void()> fill_gap,
                                        std::vector<ops_arg> &args,
                                        bool is_recomputable) {
  for (ops_arg &arg : args) {
    if (arg.argtype == OPS_ARG_DAT && arg.dat->index < 0) {
      OPSException exp(OPS_INVALID_ARGUMENT);
      exp << "ERROR: derivative dat listed as argument for external "
             "function";
      throw exp;
    }
  }
  if (!instance->ad_instance) {
    ext_func();
    return;
  }
  assert(instance->ad_instance->get_current_tape()->dagState <=
             OPSDagState::FORWARD_SAVE &&
         "INTERNAL ERROR: ops_execute_external_function reached and the tape "
         "is not in FORWARD mode");
  OPSDagState original_state =
      instance->ad_instance->get_current_tape()->dagState;
  instance->ad_instance->get_current_tape()->dagState = OPSDagState::EXTERNAL;
  instance->ad_instance->get_current_tape()->register_args(
      instance, args.data(), args.size());
  // constructor calls the ext_func as well
  instance->ad_instance->get_current_tape()->add_to_dag(
      std::make_unique<ops_external_function_node>(
          instance->ad_instance->get_current_tape()->dag.size(), ext_func,
          fill_gap, args, instance,
          instance->ad_instance->get_current_tape()->get_permanent_allocator(),
          is_recomputable));
  ops_external_function_node *node =
      reinterpret_cast<ops_external_function_node *>(
          instance->ad_instance->get_current_tape()->dag.back().get());
  instance->ad_instance->get_current_tape()->dagState = original_state;
  if (node->is_recomputable()) {
    node->visit_node();
  } else {
    node->cp_before();
    instance->ad_instance->get_current_tape()->dagState = OPSDagState::EXTERNAL;
    ext_func();
    node->save_after();
  }
  instance->ad_instance->get_current_tape()->dagState = original_state;
}

inline OPS_instance *get_instance(const ops_arg &arg) {
  switch (arg.argtype) {
  case OPS_ARG_DAT:
    return arg.dat->block->instance;
  case OPS_ARG_SCL:
    return ((ops_scalar)arg.data)->instance;
  case OPS_ARG_GBL:
    return ((ops_reduction)arg.data)->instance;
  default:
    OPSException ex(OPS_RUNTIME_ERROR);
    ex << "ops_execute_external_function called with listed argument other "
          "than ops_dat, ops_scalar or ops_reduction";
    throw ex;
    return nullptr;
  }
}

void ops_execute_external_function_impl(std::function<void()> ext_func,
                                        std::function<void()> fill_gap,
                                        std::vector<ops_arg> args,
                                        bool is_recomputable) {
  ops_execute_external_function_impl(get_instance(args[0]), ext_func, fill_gap,
                                     args, is_recomputable);
}

/////////////////////////////////////////////////////////////////////////
//                           Store Derivatives
/////////////////////////////////////////////////////////////////////////

/**
 * @brief Node type representing an ops_dat_store_derivative call in the DAG
 *
 * The forward action of the function is a NOOP, In the backward call it will
 * copy the derivative of the dat to a user managed place.
 */
class ops_store_derivative_node final : public ops_dag_node {
  ops_dat dat;
  char *user_data;
  ops_memspace memspace;

public:
  ops_store_derivative_node(int idx, ops_dat dat, char *data,
                            ops_memspace memspace)
      : ops_dag_node(idx), dat(dat), user_data(data), memspace(memspace) {}

  void visit_node() override {}

  void interpret_adjoint() override {
    ops_dat_fetch_derivative(dat, user_data, memspace);
  }

  std::string getName() const override {
    std::string name = "ops_store_derivative: ";
    return name + dat->name;
  }
};

//==================== Add store requests to DAG ====================
void ops_dat_store_derivative(ops_dat dat, char *data, ops_memspace memspace) {
  OPS_instance *instance = dat->block->instance;
  if (!instance->ad_instance) {
    return;
  }
  assert(instance->ad_instance->get_current_tape()->dagState <=
             OPSDagState::FORWARD_SAVE &&
         "INTERNAL ERROR: ops_dat_store_derivative reached and the tape "
         "is not in FORWARD mode");
  instance->ad_instance->get_current_tape()->add_to_dag(
      std::make_unique<ops_store_derivative_node>(
          instance->ad_instance->get_current_tape()->dag.size(), dat, data,
          memspace));
}

#endif /* ifndef OPS_ALGODIFF_DAG_NODE_HPP_INCLUDED */
