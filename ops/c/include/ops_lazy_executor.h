/** @file
  * @brief Lazy-execution chain executors and the per-loop runtime steps
  * @details
  * OPS can queue the loops of a program instead of running them at once ("lazy execution"):
  * with tiling (OPS_TILING), with a chain executor registered here, or with OPS_LAZY_QUEUE.
  * The queue is flushed by ops_execute(), which OPS calls whenever the host observes data
  * (reduction results, raw pointers, fetch/set data, halo transfers, exit ...).
  *
  * A chain executor takes over what happens at a flush. It receives the queued descriptors in
  * program order and must execute each exactly once, in an order that respects data dependences.
  * It may run any contiguous part of the chain through ops_lazy_execute_default() (the stock
  * generated code, tiled or eager), and it may run others with its own code, using the per-loop
  * runtime steps below so that halo exchanges, host/device coherence and dirty bits stay OPS's
  * business. The executor must not free descriptors: OPS does that after the executor returns.
  *
  * Typical use by a JIT executor, per group of loops it compiled:
  * @code
  *   for (k in group) ops_lazy_prepare(descs[k]);
  *   launch(group);                                   // uses desc->args[i].dat->data / data_d
  *   for (k in group) { ops_lazy_finish(descs[k]); ops_lazy_account(descs[k], seconds_k, bytes_k); }
  * @endcode
  */
#ifndef __OPS_LAZY_EXECUTOR_H
#define __OPS_LAZY_EXECUTOR_H

#include "ops_lib_core.h"
#include "ops_internal2.h"
#include "ops_jit_artifact.h"

/** Registers the executor of this OPS instance (NULL instance: the default one; NULL fn removes it). Registering enables queueing. */
void ops_register_chain_executor(OPS_instance *instance, ops_chain_executor_fn fn, void *ctx);

/** Non-zero if a chain executor is registered. */
int ops_has_chain_executor(OPS_instance *instance);

/** Queue loops even when tiling is off (also enabled by the command-line flag OPS_LAZY_QUEUE).
 *  Without an executor the queue is then flushed through ops_lazy_execute_default(). */
void ops_set_lazy_queue(OPS_instance *instance, int enable);

/** True if loops are being queued: tiling, an executor, or ops_set_lazy_queue(). */
int ops_lazy_queue_enabled(OPS_instance *instance);

/** Executes descs[0..n) the way OPS does without an executor: with the tiling plan when tiling
 *  is enabled, otherwise one loop after the other. The descriptors are not freed. */
void ops_lazy_execute_default(OPS_instance *instance, ops_kernel_descriptor **descs, int n);

/** Process-local execution range of a loop (the global range clipped to what this MPI rank owns,
 *  in OPS's local indexing). Fills local_range[2*d], local_range[2*d+1] for d < block->dims.
 *  Returns 0 if this process has nothing to do for the loop (the loop must be skipped here),
 *  non-zero otherwise. */
int ops_lazy_compute_local_range(ops_kernel_descriptor *desc, int *local_range);

/** Brings the data of the loop's arguments to the right place before it runs: host/device
 *  transfers and MPI halo exchanges. Time is booked as MPI time when OPS_DIAGS > 1. */
void ops_lazy_prepare(ops_kernel_descriptor *desc);

/** Marks the data written by the loop as modified (host/device dirty bits and MPI halo dirty bits). */
void ops_lazy_finish(ops_kernel_descriptor *desc);

/** Books one execution of the loop in OPS's per-kernel statistics (count, time, transferred
 *  bytes), if OPS_DIAGS > 1. `bytes` may come from ops_lazy_transfer_bytes(). */
void ops_lazy_account(ops_kernel_descriptor *desc, double seconds, double bytes);

/** Bytes the loop moves over its local range (OPS's own estimate, as printed by ops_timing_output). */
double ops_lazy_transfer_bytes(ops_kernel_descriptor *desc, const int *local_range);

/** Displacement of this process's part of the block in the global index space (zeros without MPI):
 *  global index = local index + disp. Fills disp[0..OPS_MAX_DIM). */
void ops_lazy_block_disp(ops_block block, int *disp);

/** Non-zero if OPS stores multi-component dats as structure of arrays. */
int ops_lazy_soa(OPS_instance *instance);

/** Layout of a dat in memory, for code that indexes dat->data (host) or dat->data_d (device) itself.
 *  The element at local loop index i along dimension d (the indexing of ops_lazy_compute_local_range) is at
 *  position i + shift[d] along d of the allocation, which has dat->size[d] elements along d; the first dimension is
 *  the fastest. This is the same offset as dat->base_offset, split per dimension: -base - d_m, and under MPI
 *  - d_im as well. Fills shift[0..OPS_MAX_DIM). */
void ops_lazy_dat_shift(ops_dat dat, int *shift);

/** Where loop `desc` accumulates the result of its reduction argument `arg` (an OPS_ARG_GBL argument with an access
 *  other than OPS_READ): dim elements of the argument's type, which hold the value so far (a reduction combines
 *  into it). Under MPI the handle keeps one slot per block. */
char *ops_lazy_reduction_slot(ops_kernel_descriptor *desc, int arg);

/** Registers a built-in executor that must not change any application's results, to test the executor
 *  interface with unmodified applications. mode: "split[,seed=N]" hands the queue to
 *  ops_lazy_execute_default() in pieces of 1 to 5 loops; "manual" runs each loop with the per-loop steps
 *  above. Selected on the command line with OPS_LAZY_EXECUTOR_TEST=<mode>. */
void ops_lazy_enable_test_executor(OPS_instance *instance, const char *mode);

/** Value of a global constant declared with ops_decl_const / ops_update_const. */
typedef struct {
  const char *name;
  const char *type;     /**< type name as given to ops_decl_const, e.g. "double" or "field_type" */
  int dim;              /**< number of elements */
  int size;             /**< bytes per element */
  const char *snapshot; /**< dim*size bytes: the value as of the last declaration/update; owned by OPS */
  unsigned version;     /**< incremented whenever the value changes */
} ops_const_entry;

/** The constant `name`, or NULL if it was never declared. The pointer (and the snapshot) stay valid
 *  until the constant changes again.
 *
 *  Constants must be changed through ops_update_const()/ops_decl_const() (the call may follow the
 *  assignment to the variable). If the value changes while loops are queued, those loops are executed
 *  before the new value is recorded, so a chain executor that reads the snapshot sees, for every
 *  queued loop, the value the loop was enqueued with. (Stock generated CPU code reads the live
 *  variable when the loop finally runs; only code that uses the snapshot gets enqueue-time values.) */
const ops_const_entry *ops_find_const(OPS_instance *instance, const char *name);

#endif /* __OPS_LAZY_EXECUTOR_H */
