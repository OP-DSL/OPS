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

/** Registers the executor of this OPS instance (NULL removes it). Registering enables queueing. */
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

#endif /* __OPS_LAZY_EXECUTOR_H */
