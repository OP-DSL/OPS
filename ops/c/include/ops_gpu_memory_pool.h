#ifndef __OPS_GPU_MEMORY_POOL_H
#define __OPS_GPU_MEMORY_POOL_H

#include <stddef.h>
#include <vector>
#include <map>
#include <mutex>
#include <ops_lib_core.h>

#ifdef __cplusplus
extern "C" {
#endif


// Memory block structure to track allocations
struct ops_memory_block {
    void* ptr;          // Pointer to the memory block
    size_t size;        // Size of the block
    size_t offset;      // Offset from the pool base address
    bool is_free;       // Whether the block is free
    
    ops_memory_block(void* p, size_t s, size_t o, bool free = true) 
        : ptr(p), size(s), offset(o), is_free(free) {}
};

// GPU Memory Pool class
class OPSGPUMemoryPool {
private:
    void* pool_base;                               // Base address of the memory pool
    size_t pool_size;                             // Total size of the memory pool
    size_t allocated_size;                        // Currently allocated size
    std::vector<ops_memory_block> memory_blocks;  // List of memory blocks
    std::map<void*, size_t> pointer_to_block;    // Map from pointer to block index
    std::mutex pool_mutex;                        // Thread safety
    bool pool_initialized;                        // Whether the pool has been initialized
    
    // Platform-specific function pointers
    int (*platform_malloc)(void** ptr, size_t bytes);
    int (*platform_free)(void* ptr);
    int (*platform_meminfo)(size_t* free, size_t* total);
    
public:
    OPSGPUMemoryPool();
    ~OPSGPUMemoryPool();
    
    // Initialize the memory pool with platform-specific functions
    void initialize(
        int (*malloc_func)(void** ptr, size_t bytes),
        int (*free_func)(void* ptr),
        int (*meminfo_func)(size_t* free, size_t* total),
        double memory_fraction = 0.9
    );
    
    // Allocate memory from the pool
    void* allocate(size_t bytes);
    
    // Free memory back to the pool
    void deallocate(void* ptr);
    
    // Get pool statistics
    void get_stats(size_t* total_size, size_t* allocated_size, size_t* free_size);
    
    // Check if the pool is initialized
    bool is_initialized() const { return pool_initialized; }
    
    // Reset/cleanup the pool
    void cleanup();
    
private:
    // Internal helper functions
    void coalesce_free_blocks();
    size_t find_free_block(size_t size);
    void split_block(size_t block_index, size_t size);
};

// Global memory pool instance (singleton)
extern OPSGPUMemoryPool* ops_gpu_memory_pool;

// C interface functions
void ops_gpu_memory_pool_init(
    int (*malloc_func)(void** ptr, size_t bytes),
    int (*free_func)(void* ptr),
    int (*meminfo_func)(size_t* free, size_t* total),
    double memory_fraction
);

void* ops_gpu_memory_pool_alloc(size_t bytes);
void ops_gpu_memory_pool_free(void* ptr);
void ops_gpu_memory_pool_stats(size_t* total_size, size_t* allocated_size, size_t* free_size);
void ops_gpu_memory_pool_cleanup();

#ifdef __cplusplus
}
#endif

#endif // __OPS_GPU_MEMORY_POOL_H 
