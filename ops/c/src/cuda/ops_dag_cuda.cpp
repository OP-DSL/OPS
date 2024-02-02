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

/** @brief OPS DAG traversing capabilities
 * @author Gabor Daniel Balogh
 * @details Implementations of the core library functions utilized by all OPS
 * backends
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ops_cuda_rt_support.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "ops_algodiff.hpp"
#include "ops_lib_core.h"

/////////////////////////////////////////////////////////////////////////
// Copy implementations
/////////////////////////////////////////////////////////////////////////

#define MY_COPY(dest, orig, N) \
  cudaMemcpy((void *)dest, (void *)orig, N, cudaMemcpyDeviceToDevice)
// #define MY_COPY(dest, orig, N)

/////////////////////////////////////////////////////////////////////////
// Data structures
/////////////////////////////////////////////////////////////////////////

char *ReusableMonotonicAllocator::allocate(size_t chunkSize) {
  auto tmp = std::chrono::high_resolution_clock::now();
  if (chunkSize % sizeof(double))
    chunkSize += sizeof(double) - chunkSize % sizeof(double);
  if (!head) {
    assert(head_idx == 0);
    size_t lastPoolSize = std::min(std::max(minChunkSize, 10 * chunkSize),
                                   maxChunkSize - maxChunkSize % chunkSize);
    if (chunkSize > maxChunkSize)
      lastPoolSize = chunkSize;
    cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),
                  cudaMalloc((void **)&head, lastPoolSize));
    chunks.emplace_back(lastPoolSize, head);
  } else if (head + chunkSize >
             chunks[head_idx].second + chunks[head_idx].first) {
    // current chunk is too small, search for the first that can hold this chunk
    while (++head_idx < chunks.size() && chunks[head_idx].first < chunkSize) {
      // do nothing
    }
    if (head_idx == chunks.size()) {
      size_t lastPoolSize = 2 * chunks[head_idx - 1].first;
      if (chunkSize > maxChunkSize) {
        lastPoolSize = chunkSize;
      } else {
        lastPoolSize = std::max(10 * chunkSize, lastPoolSize);
        lastPoolSize =
            std::min(lastPoolSize, maxChunkSize - maxChunkSize % chunkSize);
      }
      cutilSafeCall(OPS_instance::getOPSInstance()->ostream(),
                    cudaMalloc((void **)&head, lastPoolSize));
      chunks.emplace_back(lastPoolSize, head);
    }
    head = chunks[head_idx].second;
  }
  overhead += std::chrono::high_resolution_clock::now() - tmp;
  // return std::exchange(head, head + chunkSize);
  char *oldhead = head;
  head = head + chunkSize;
  return oldhead;
}

void ReusableMonotonicAllocator::release() {
  for (auto &chunk : chunks) {
    // _mm_free(chunk.second);
    cudaFree(chunk.second);
  }
  chunks.clear();
  head = nullptr;
  head_idx = 0;
}

void ReusableMonotonicAllocator::free(char *) {
  // No-op for now
}

void ReusableMonotonicAllocator::free_all() {
  if (!head) return;
  head_idx = 0;
  head = chunks[0].second;
}

ReusableMonotonicAllocator::~ReusableMonotonicAllocator() {
  std::cout << "  TIMING: MonotonicAllocator_overhead = " << overhead.count()
            << " ms\n";
  release();
}

char *CheckpointAllocator::allocate(size_t chunkSize) {
  auto tmp = std::chrono::high_resolution_clock::now();
  if (chunkSize % sizeof(double))
    chunkSize += sizeof(double) - chunkSize % sizeof(double);

  // search for available free checkpoint
  for (int i = free_chunks.size() - 1; i >= 0; --i) {
    if (free_chunks[i].first >= chunkSize) {
      chunks.push_back(free_chunks[i]);
      std::swap(free_chunks.back(), free_chunks[i]);
      free_chunks.pop_back();
      // TODO @bgd fix overhead measure
      overhead += std::chrono::high_resolution_clock::now() - tmp;
      return chunks.back().second;
    }
  }

  // if no free chunk then create one
  chunks.emplace_back(chunkSize, upstream.allocate(chunkSize));

  // TODO @bgd fix overhead measure
  overhead += std::chrono::high_resolution_clock::now() - tmp;
  return chunks.back().second;
}

void CheckpointAllocator::release() {
  chunks.clear();
  free_chunks.clear();
  upstream.release();
}

void CheckpointAllocator::free(char *cp) {
  for (int i = chunks.size() - 1; i >= 0; --i) {
    if (chunks[i].second == cp) {
      free_chunks.push_back(chunks[i]);
      std::swap(chunks.back(), chunks[i]);
      chunks.pop_back();
      return;
    }
  }
  assert(false);
}

CheckpointAllocator::~CheckpointAllocator() {
  std::cout << "  TIMING: CPAllocator_overhead = " << overhead.count()
            << " ms\n";
  release();
}

/////////////////////////////////////////////////////////////////////////
// Checkpoint allocation and restoration for ops_dat
/////////////////////////////////////////////////////////////////////////

void ops_copy_cp(ops_dat dat, const char *src) {
  MY_COPY(dat->data_d, src, dat->mem);
}
void ops_copy_cp_to_buf(ops_dat dat, const char *buf) {
  MY_COPY(buf, dat->data_d, dat->mem);
}

void ops_fetch_derivative(const ops_dat dat, char *data) {
  if (dat->is_passive) {
    std::string err = "Request derivative of a passive ops_dat(";
    err += dat->name;
    err += ")";
    throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
  }
  cudaMemcpy((void *)data, (void *)dat->derivative, dat->mem,
             cudaMemcpyDeviceToHost);
}

void ops_set_derivative(const ops_dat dat, char *data) {
  if (dat->is_passive) {
    std::string err = "Write derivative of a passive ops_dat(";
    err += dat->name;
    err += ")";
    throw OPSException(OPS_RUNTIME_ERROR, err.c_str());
  }
  cudaMemcpy((void *)dat->derivative, (void *)data, dat->mem,
             cudaMemcpyHostToDevice);
}

// Hack to avoid the linker to drop allocator functions
char *ops_alloc_cp(ops_dat dat, int size) {
  return dat->block->instance->ad_instance->get_current_tape()->alloc_cp(dat,
                                                                         size);
}

char *ops_ad_tape::alloc_cp(ops_dat dat, int size) {
  assert(dats_used.size() > static_cast<size_t>(dat->index));
  assert(dats_used[dat->index]);
  return dats_used[dat->index]->alloc_cp(size, current_tape_idx);
}

char *ops_ad_dat_state::alloc_cp(int bytes, int tape_idx) {
  assert(dat_states.size() == 0 || tape_idx > dat_states.back().first);
  dat_states.push_back({tape_idx, allocator->allocate(bytes)});
  cp_idx = dat_states.size();
  return dat_states.back().second;
}

#undef MY_COPY
