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

/** @brief ops checkpointing library function declarations
  * @author Istvan Reguly
  * @details function declarations for checkpointing
  */

#ifndef __OPS_CHECKPOINTING_H
#define __OPS_CHECKPOINTING_H

typedef enum {OPS_BACKUP_GATHER, OPS_BACKUP_LEADIN, OPS_BACKUP_RESTORE, OPS_BACKUP_BEGIN, OPS_BACKUP_IN_PROCESS, OPS_BACKUP_END, OPS_NONE} ops_backup_state;
typedef enum {OPS_NOT_SAVED, OPS_SAVED, OPS_UNDECIDED } ops_checkpoint_types;
typedef enum {OPS_CHECKPOINT_INITPHASE=1, OPS_CHECKPOINT_MANUAL_DATLIST=2, OPS_CHECKPOINT_FASTFW=4} ops_checkpoint_options;

#ifdef __cplusplus
extern "C" {
#endif

bool ops_checkpointing_init(const char *filename, double interval, int options);
void ops_checkpointing_initphase_done();
bool ops_checkpointing_before(ops_arg *args, int nargs, int *range, int loop_id);
bool ops_checkpointing_name_before(ops_arg *args, int nargs, int *range, const char *s);
void ops_checkpointing_exit();
void ops_checkpointing_reduction(ops_reduction red);
void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist);
bool ops_checkpointing_fastfw(int nbytes, char *payload);
bool ops_checkpointing_manual_datlist_fastfw(int ndats, ops_dat *datlist, int nbytes, char *payload);

extern ops_backup_state backup_state;
extern char *OPS_dat_ever_written;
extern ops_checkpoint_types *OPS_dat_status;
extern int OPS_ranks_per_node;

#ifdef __cplusplus
}
#endif

#endif/* __OPS_CHECKPOINTING_H */
