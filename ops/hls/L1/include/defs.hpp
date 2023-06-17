
#pragma once

/* data mover limits */
constexpr unsigned int max_data_size = 4000000000;
constexpr unsigned int avg_data_size = 4244832; //100 x 100 x 100 grid with (-1,+1) halo with 4-byte datatype in bytes

constexpr unsigned int min_mem_data_width = 128;
constexpr unsigned int avg_mem_data_width = 512;
constexpr unsigned int max_mem_data_width = 1024;
constexpr unsigned int min_stream_data_width = 64;
constexpr unsigned int avg_stream_data_width = 256;
constexpr unsigned int max_stream_data_width = 1024;
constexpr unsigned int min_burst_len = 1;
constexpr unsigned int avg_burst_len = 32;
constexpr unsigned int max_burst_len = 64;
constexpr unsigned int max_strm_pkts_per_beat = max_mem_data_width / min_stream_data_width;
constexpr unsigned int avg_strm_pkts_per_beat = avg_mem_data_width / avg_stream_data_width;
constexpr unsigned int min_strm_pkts_per_beat = 1;

constexpr unsigned int min_bytes_per_beat = min_mem_data_width / 8;
constexpr unsigned int avg_bytes_per_beat = avg_mem_data_width / 8;
constexpr unsigned int avg_num_of_bursts = (avg_data_size + avg_bytes_per_beat - 1) / avg_bytes_per_beat;
constexpr unsigned int max_num_of_bursts = (max_data_size + min_bytes_per_beat - 1) / min_bytes_per_beat;
