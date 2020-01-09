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
  * @brief OPS checkpointing library function declarations
  * @author Istvan Reguly
  * @details function declarations for checkpointing
  */

#ifndef __OPS_CHECKPOINTING_H
#define __OPS_CHECKPOINTING_H

typedef enum {
  OPS_BACKUP_GATHER,
  OPS_BACKUP_LEADIN,
  OPS_BACKUP_RESTORE,
  OPS_BACKUP_BEGIN,
  OPS_BACKUP_IN_PROCESS,
  OPS_BACKUP_END,
  OPS_NONE
} ops_backup_state;
typedef enum { OPS_NOT_SAVED, OPS_SAVED, OPS_UNDECIDED } ops_checkpoint_types;

typedef enum {
    /**
     * Indicates that there are a number of parallel loops at the very beginning
     * of the simulations which should be excluded from any checkpoint;
     * mainly because they initialise datasets that do not change during the
     * main body of the execution.
     * During restore mode these loops are executed as usual.
     * An example would be the computation of the mesh geometry, which can be
     * excluded from the checkpoint if it is re-computed when recovering and
     * restoring a checkpoint.
     * The API call void ops_checkpointing_initphase_done() indicates the end
     * of this initial phase.
     */
  OPS_CHECKPOINT_INITPHASE = 1,
    /**
     * Indicates that the user manually controls the location of the checkpoint,
     * and explicitly specifies the list of ::ops_dat s to be saved.
     */
  OPS_CHECKPOINT_MANUAL_DATLIST = 2,
    /**
     * Indicates that the user manually controls the location of the checkpoint,
     * and it also enables fast-forwarding, by skipping the execution of the
     * application (even though none of the parallel loops would actually
     * execute, there may be significant work outside of those) up to the
     * checkpoint.
     */
  OPS_CHECKPOINT_FASTFW = 4,
    /**
     * Indicates that when the corresponding API function is called, the
     * checkpoint should be created.
     * Assumes the presence of the above two options as well.
     */
  OPS_CHECKPOINT_MANUAL = 8
} ops_checkpoint_options;

#ifndef OPS_CPP_API
/**
 * Initialises the checkpointing system, has to be called after ops_partition().
 *
 * @param filename  name of the file for checkpointing.
 *                In MPI, this will automatically be postfixed with the rank ID.
 * @param interval  average time (seconds) between checkpoints
 * @param options   a combinations of flags, defined by ::ops_checkpoint_options
 * @return          `true` if the application launches in restore mode,
 *                  `false` otherwise.
 */
OPS_FTN_INTEROP
bool ops_checkpointing_init(const char *filename, double interval, int options);
OPS_FTN_INTEROP
void ops_checkpointing_initphase_done();
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range,
                              int loop_id);
bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range,
                                   const char *s);
void ops_checkpointing_exit(OPS_instance *);
void ops_checkpointing_reduction(ops_reduction red);

/**
 * Call this routine at a point in the code to mark the location of a checkpoint.
 *
 * At this point, the list of datasets specified will be saved.
 * The validity of what is saved is not checked by the checkpointing algorithm
 * assuming that the user knows what data sets to be saved for full recovery.
 * This routine should be called frequently (compared to check-pointing
 * frequency) and it will trigger the creation of the checkpoint the first time
 * it is called after the timeout occurs.
 *
 * @param ndats    number of datasets to be saved
 * @param datlist  arrays of ::ops_dat handles to be saved
 */
OPS_FTN_INTEROP
void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist);

/**
 * Call this routine at a point in the code to mark the location of a checkpoint.
 *
 * At this point, the specified payload (e.g. iteration count, simulation time,
 * etc.) along with the necessary datasets, as determined by the checkpointing
 * algorithm will be saved.
 * This routine should be called frequently (compared to checkpointing
 * frequency), will trigger the creation of the checkpoint the first time it is
 * called after the timeout occurs.
 * In restore mode, will restore all datasets the first time it is called, and
 * returns `true` indicating that the saved payload is returned in payload.
 * Does not save reduction data.
 *
 * @param nbytes   size of the payload in bytes
 * @param payload  pointer to memory into which the payload is packed
 * @return
 */
OPS_FTN_INTEROP
bool ops_checkpointing_fastfw(int nbytes, char *payload);

/**
 * Combines the ops_checkpointing_manual_datlist() and
 * ops_checkpointing_fastfw() calls.
 *
 * @param ndats    number of datasets to be saved
 * @param datlist  arrays of ::ops_dat handles to be saved
 * @param nbytes   size of the payload in bytes
 * @param payload  pointer to memory into which the payload is packed
 * @return
 */
OPS_FTN_INTEROP
bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist,
                                             int nbytes, char *payload);

/**
 * With this routine it is possible to manually trigger checkpointing,
 * instead of relying on the timeout process.
 *
 * It combines the ops_checkpointing_manual_datlist() and
 * ops_checkpointing_fastfw() calls, and triggers the creation of a
 * checkpoint when called.
 *
 * @param ndats    number of datasets to be saved
 * @param datlist  arrays of ::ops_dat handles to be saved
 * @param nbytes   size of the payload in bytes
 * @param payload  pointer to memory into which the payload is packed
 * @return
 */
OPS_FTN_INTEROP
bool ops_checkpointing_manual_datlist_fastfw_trigger(int ndats,
                                                     ops_dat *datlist,
                                                     int nbytes, char *payload);
#endif /* OPS_CPP_API */
#endif /* __OPS_CHECKPOINTING_H */
