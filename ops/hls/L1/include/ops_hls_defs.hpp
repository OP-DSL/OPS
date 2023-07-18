
#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @file
  * @brief HLS related definitions.
  * @author Beniel Thileepan
  * @details Contains limits and constrains definitions used in L1 components.
  */

/***** OPS limits *****/
constexpr unsigned int ops_max_dim = 3;

/* Tiling */
constexpr unsigned int max_depth_bytes = 8192;
constexpr unsigned int max_depth_v8 = max_depth_bytes / 8;
constexpr unsigned int max_depth_v16 = max_depth_bytes / 16;
constexpr unsigned int max_depth_d32 = max_depth_bytes / 4;
constexpr unsigned int max_depth_d32_v8 = max_depth_d32 / 8;
constexpr unsigned int max_depth_d32_v16 = max_depth_d32 / 16;

constexpr unsigned int min_depth_bytes = 32 * 4;
constexpr unsigned int min_depth_d32 = min_depth_bytes / 4;
constexpr unsigned int min_depth_d32_v8 = min_depth_d32 / 8;
constexpr unsigned int min_depth_d32_v16 = min_depth_d32 / 16;

constexpr unsigned int avg_depth_bytes = 1000 * 4;
constexpr unsigned int avg_depth_d32 = avg_depth_bytes / 4;
constexpr unsigned int avg_depth_d32_v8 = avg_depth_d32 / 8;
constexpr unsigned int avg_depth_d32_v16 = avg_depth_d32 / 16;

/* Grid limits */
constexpr unsigned int max_blocks = max_depth_d32_v8 + 1;
constexpr unsigned int min_blocks = min_depth_d32_v16 + 1;
constexpr unsigned int avg_blocks = avg_depth_d32_v8 + 1;

constexpr unsigned int max_grid_size = max_blocks * max_depth_d32 * max_depth_d32;
constexpr unsigned int min_grid_size = min_blocks * min_depth_d32 * min_depth_d32;
constexpr unsigned int avg_grid_size = avg_blocks * avg_depth_d32 * avg_depth_d32;

/* data mover limits */
constexpr unsigned int max_data_size = 4000000000;
constexpr unsigned int avg_data_size = 4244832; //100 x 100 x 100 grid with (-1,+1) halo with 4-byte datatype in bytes

constexpr unsigned int min_mem_data_width = 128;
constexpr unsigned int avg_mem_data_width = 512;
constexpr unsigned int max_mem_data_width = 1024;

constexpr unsigned int min_axis_data_width = 64;
constexpr unsigned int avg_axis_data_width = 256;
constexpr unsigned int max_axis_data_width = 1024;

constexpr unsigned int min_hls_stream_data_width = 64;
constexpr unsigned int avg_hls_stream_data_width = 256;
constexpr unsigned int max_hls_stream_data_width = 1024;

constexpr unsigned int min_burst_len = 1;
constexpr unsigned int avg_burst_len = 32;
constexpr unsigned int max_burst_len = 64;

constexpr unsigned int max_strm_pkts_per_beat = max_mem_data_width / min_axis_data_width;
constexpr unsigned int avg_strm_pkts_per_beat = avg_mem_data_width / avg_axis_data_width;
constexpr unsigned int min_strm_pkts_per_beat = 1;

constexpr unsigned int min_bytes_per_beat = min_mem_data_width / 8;
constexpr unsigned int avg_bytes_per_beat = avg_mem_data_width / 8;
constexpr unsigned int avg_num_of_bursts = (avg_data_size + avg_bytes_per_beat - 1) / avg_bytes_per_beat;
constexpr unsigned int max_num_of_bursts = (max_data_size + min_bytes_per_beat - 1) / min_bytes_per_beat;

namespace ops {
namespace hls {

//Float - Int convertor
typedef union 
{
    unsigned int i;
    float f;
} DataConv;


typedef unsigned short SizeType[ops_max_dim];


struct GridPropertyCore
{
    SizeType size;
    SizeType d_p;
    SizeType d_m;
    SizeType grid_size;
    SizeType actual_size;
    SizeType offset;
    unsigned short dim;
    unsigned short xblocks;
    unsigned int total_itr;
    unsigned outer_loop_limit;
};

enum CoefTypes
{
    CONST_COEF=0,
    DYNAMIC_COEF=1
};


}
}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */
