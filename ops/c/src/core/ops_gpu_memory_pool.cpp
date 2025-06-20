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
* notice, this list of conditions and the disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the disclaimer in the
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
  * @brief Shared GPU Memory Pool Implementation
  * @author OPS Development Team
  * @details Implements a memory pool for efficient GPU memory management
  * that works with both CUDA and HIP backends
  */

#include "ops_gpu_memory_pool.h"
#include <iostream>
#include <algorithm>
#include <cassert>

// Global memory pool instance
OPSGPUMemoryPool* ops_gpu_memory_pool = nullptr;

// Constructor
OPSGPUMemoryPool::OPSGPUMemoryPool() 
    : pool_base(nullptr), pool_size(0), allocated_size(0), 
      pool_initialized(false), platform_malloc(nullptr),
      platform_free(nullptr), platform_meminfo(nullptr) {
}

// Destructor
OPSGPUMemoryPool::~OPSGPUMemoryPool() {
    cleanup();
}

// Initialize the memory pool
void OPSGPUMemoryPool::initialize(
    int (*malloc_func)(void** ptr, size_t bytes),
    int (*free_func)(void* ptr),
    int (*meminfo_func)(size_t* free, size_t* total),
    double memory_fraction
) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (pool_initialized) {
        std::cerr << "Warning: GPU memory pool already initialized\n";
        return;
    }
    
    // Store platform-specific function pointers
    platform_malloc = malloc_func;
    platform_free = free_func;
    platform_meminfo = meminfo_func;
    
    // Get available GPU memory
    size_t free_mem, total_mem;
    if (platform_meminfo(&free_mem, &total_mem) != 0) {
        std::cerr << "Error: Failed to query GPU memory information\n";
        return;
    }
    
    // Calculate pool size (fraction of available memory)
    pool_size = static_cast<size_t>(free_mem * memory_fraction);
    
    // Align to 256-byte boundary for better performance
    pool_size = (pool_size / 256) * 256;
    
    // Allocate the memory pool
    if (platform_malloc(&pool_base, pool_size) != 0) {
        std::cerr << "Error: Failed to allocate GPU memory pool of size " 
                  << pool_size << " bytes\n";
        return;
    }
    
    // Initialize with one large free block
    memory_blocks.clear();
    memory_blocks.emplace_back(pool_base, pool_size, 0, true);
    pointer_to_block.clear();
    allocated_size = 0;
    pool_initialized = true;
    
    std::cout << "GPU Memory Pool initialized: " << pool_size / (1024*1024) 
              << " MB (" << (memory_fraction * 100) << "% of " 
              << total_mem / (1024*1024) << " MB total)\n";
}

// Allocate memory from the pool
void* OPSGPUMemoryPool::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (!pool_initialized) {
        std::cerr << "Error: GPU memory pool not initialized\n";
        return nullptr;
    }
    
    // Align allocation to 256-byte boundary
    size_t aligned_bytes = ((bytes + 255) / 256) * 256;
    
    // Find a suitable free block
    size_t block_index = find_free_block(aligned_bytes);
    if (block_index == SIZE_MAX) {
        // Try to coalesce free blocks and search again
        coalesce_free_blocks();
        block_index = find_free_block(aligned_bytes);
        
        if (block_index == SIZE_MAX) {
            std::cerr << "Error: GPU memory pool out of memory. "
                      << "Requested: " << bytes << " bytes, "
                      << "Available: " << (pool_size - allocated_size) << " bytes\n";
            return nullptr;
        }
    }
    
    // Split the block if it's larger than needed
    if (memory_blocks[block_index].size > aligned_bytes) {
        split_block(block_index, aligned_bytes);
    }
    
    // Mark the block as allocated
    memory_blocks[block_index].is_free = false;
    allocated_size += memory_blocks[block_index].size;
    
    // Add to pointer map
    pointer_to_block[memory_blocks[block_index].ptr] = block_index;
    
    return memory_blocks[block_index].ptr;
}

// Free memory back to the pool
void OPSGPUMemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (!pool_initialized || ptr == nullptr) {
        return;
    }
    
    // Find the block
    auto it = pointer_to_block.find(ptr);
    if (it == pointer_to_block.end()) {
        std::cerr << "Error: Attempting to free invalid pointer " << ptr 
                  << " (not found in pool allocation map)\n";
        return;
    }
    
    size_t block_index = it->second;
    
    // Validate that the block index is still valid
    if (block_index >= memory_blocks.size()) {
        std::cerr << "Error: Invalid block index " << block_index 
                  << " (vector size: " << memory_blocks.size() << ")\n";
        return;
    }
    
    if (memory_blocks[block_index].is_free) {
        std::cerr << "Error: Double free detected for pointer " << ptr << "\n";
        return;
    }
    
    // Additional validation: check that the pointer matches
    if (memory_blocks[block_index].ptr != ptr) {
        std::cerr << "Error: Pointer mismatch in block " << block_index 
                  << " (expected: " << memory_blocks[block_index].ptr 
                  << ", got: " << ptr << ")\n";
        return;
    }
    
    // Mark the block as free
    memory_blocks[block_index].is_free = true;
    allocated_size -= memory_blocks[block_index].size;
    
    // Remove from pointer map
    pointer_to_block.erase(it);
    
    // Coalesce adjacent free blocks
    coalesce_free_blocks();
}

// Get pool statistics
void OPSGPUMemoryPool::get_stats(size_t* total_size, size_t* allocated_size_out, size_t* free_size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (total_size) *total_size = pool_size;
    if (allocated_size_out) *allocated_size_out = allocated_size;
    if (free_size) *free_size = pool_size - allocated_size;
}

// Cleanup the pool
void OPSGPUMemoryPool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    if (pool_initialized && pool_base && platform_free) {
        platform_free(pool_base);
        pool_base = nullptr;
        pool_size = 0;
        allocated_size = 0;
        memory_blocks.clear();
        pointer_to_block.clear();
        pool_initialized = false;
    }
}

// Coalesce adjacent free blocks
void OPSGPUMemoryPool::coalesce_free_blocks() {
    // Sort blocks by offset to make coalescing easier
    std::sort(memory_blocks.begin(), memory_blocks.end(),
              [](const ops_memory_block& a, const ops_memory_block& b) {
                  return a.offset < b.offset;
              });
    
    // Coalesce adjacent free blocks
    for (size_t i = 0; i < memory_blocks.size(); ) {
        if (!memory_blocks[i].is_free) {
            ++i;
            continue;
        }
        
        // Look for adjacent free blocks
        size_t j = i + 1;
        while (j < memory_blocks.size() && 
               memory_blocks[j].is_free &&
               memory_blocks[i].offset + memory_blocks[i].size == memory_blocks[j].offset) {
            // Merge block j into block i
            memory_blocks[i].size += memory_blocks[j].size;
            ++j;
        }
        
        // Remove merged blocks
        if (j > i + 1) {
            memory_blocks.erase(memory_blocks.begin() + i + 1, memory_blocks.begin() + j);
        }
        
        ++i;
    }
    
    // Rebuild pointer map AFTER coalescing to ensure correct indices
    pointer_to_block.clear();
    for (size_t i = 0; i < memory_blocks.size(); ++i) {
        if (!memory_blocks[i].is_free) {
            pointer_to_block[memory_blocks[i].ptr] = i;
        }
    }
}

// Find a free block of at least the given size
size_t OPSGPUMemoryPool::find_free_block(size_t size) {
    for (size_t i = 0; i < memory_blocks.size(); ++i) {
        if (memory_blocks[i].is_free && memory_blocks[i].size >= size) {
            return i;
        }
    }
    return SIZE_MAX; // Not found
}

// Split a block into two parts
void OPSGPUMemoryPool::split_block(size_t block_index, size_t size) {
    ops_memory_block& block = memory_blocks[block_index];
    
    if (block.size <= size) {
        return; // No need to split
    }
    
    // Create a new block for the remaining space
    size_t remaining_size = block.size - size;
    size_t new_offset = block.offset + size;
    void* new_ptr = static_cast<char*>(pool_base) + new_offset;
    
    memory_blocks.emplace_back(new_ptr, remaining_size, new_offset, true);
    
    // Update the original block
    block.size = size;
}

// C interface functions
void ops_gpu_memory_pool_init(
    int (*malloc_func)(void** ptr, size_t bytes),
    int (*free_func)(void* ptr),
    int (*meminfo_func)(size_t* free, size_t* total),
    double memory_fraction
) {
    if (!ops_gpu_memory_pool) {
        ops_gpu_memory_pool = new OPSGPUMemoryPool();
    }
    ops_gpu_memory_pool->initialize(malloc_func, free_func, meminfo_func, memory_fraction);
}

void* ops_gpu_memory_pool_alloc(size_t bytes) {
    if (!ops_gpu_memory_pool) {
        return nullptr;
    }
    return ops_gpu_memory_pool->allocate(bytes);
}

void ops_gpu_memory_pool_free(void* ptr) {
    if (ops_gpu_memory_pool) {
        ops_gpu_memory_pool->deallocate(ptr);
    }
}

void ops_gpu_memory_pool_stats(size_t* total_size, size_t* allocated_size, size_t* free_size) {
    if (ops_gpu_memory_pool) {
        ops_gpu_memory_pool->get_stats(total_size, allocated_size, free_size);
    }
}

void ops_gpu_memory_pool_cleanup() {
    if (ops_gpu_memory_pool) {
        ops_gpu_memory_pool->cleanup();
        delete ops_gpu_memory_pool;
        ops_gpu_memory_pool = nullptr;
    }
} 