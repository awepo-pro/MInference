// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <cuda.h>

// __device__ int min(int x, int y) {
//     return x < y ? x : y;
// }

// __device__ int max(int x, int y) {
//     return x > y ? x : y;
// }

__device__ void save_blocks(int* block_offset, int range_start, int range_end, int block_size, int& block_count) {
    // * current block `block_count` to compute range of `idx` 
    for (int idx = range_start; idx < range_end; idx += block_size) {
        block_offset[block_count++] = idx;
    }
}

__global__ void convert_vertical_slash_indexes_kernel(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S], NNZ_S := slash_indexes.size()
    int* column_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V], NNZ_V := vertical_indexes.size()
    int N_HEADS,
    int N_ROWS,
    int BLOCK_SIZE_M,             // 64 constexpr
    int BLOCK_SIZE_N,             // 64 constexpr
    int NNZ_V,
    int NNZ_S
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    // * group := number of row process in 1 thread
    const int group_idx = blockIdx.z;

    // TODO: seqlens' dim should be 1 * 1 
    int seqlen = seqlens[batch_idx];
    // * expand from z-axis
    int block_idx_m = group_idx * blockDim.x + threadIdx.x;

    // * starting point of seq that the thread is required to process
    int start_m = block_idx_m * BLOCK_SIZE_M;
    if (start_m >= seqlen) 
        return;
    
    int end_m = start_m + BLOCK_SIZE_M;
    // * moving pointer! `const int *` mean the (deference) data is const, doesn't mean cannot move pointer
    // * (batch_idx * N_HEADS + head_idx) -> head's position, NNZ_V := dim of seq. The increment means moving to corresponding head, given batch's idx
    
    // * to visualize it, (batch_idx * N_HEADS) moving up in y-asix, +head_dix moving right in x-axis, NNZ_V moving toward user in z-axis
    vertical_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_V;
    slash_indexes += (batch_idx * N_HEADS + head_idx) * NNZ_S;

    int row_offset = (batch_idx * N_HEADS + head_idx) * N_ROWS + block_idx_m;
    block_count += row_offset;
    block_offset += row_offset * NNZ_S;
    column_count += row_offset;
    column_index += row_offset * NNZ_V;

    int tmp_col_cnt = 0, tmp_blk_cnt = 0;
    int s = 0, v = 0;
    int v_idx = vertical_indexes[v++];
    int s_idx = slash_indexes[s++];
    
    // * ================= find range of slash ============================
    // * s_idx := how many tokens (headdim element) I can see from current block
    while (s_idx >= end_m) 
        s_idx = slash_indexes[s++];

    // * ie. end_m = 16, s_idx = 12 
    // *    - it implies element I can see is 4, from [12, 16). 
    // *    - also, since it is slash (rmb that we compute accumulation in slash), that means in next row can see [11, 15), next next row see [10, 14)
    // *    - if we accumulate BLOCK_SIZE_M such element, last row can see [12 - BLOCK_SIZE_M, 16 - BLOCK_SIZE_M). Its lower bound must be at least BLOCK_SIZE_M
    // *    - after that, we find `range_start = s_idx - blokck_size_M` as expected; now, we do alignment, the range_end != end_m
    s_idx = max(end_m - s_idx, BLOCK_SIZE_M);
    int range_start = s_idx - BLOCK_SIZE_M, range_end = s_idx;
    // * =================== end =====================================

    while (1) {
        if (v_idx < range_end) {
            // * if vertice index is not in slash range, add it independently
            if (v_idx < range_start) {
                column_index[tmp_col_cnt++] = v_idx;
            }
            // * if v still in vertical_indexex range
            if (v < NNZ_V) {
                v_idx = vertical_indexes[v++];
            } else {
                // * TODO: why add BLOCK_SIZE_M?
                v_idx = end_m + BLOCK_SIZE_M;
            }
        } else {
            if (s < NNZ_S) {
                s_idx = max(end_m - slash_indexes[s++], BLOCK_SIZE_M);
            } else {
                // * save_blocks only record the lower bound. ie. [12, 16) -> record block_offset[blk_cnt++] = 12; 
                // * However, if the range is too large [0, 100) and block_size_N = 20, we might chunk the range into 5 partitions. 
                // *    - block_offset[0] = 0; block_offset[1] = 20; block_offset[2] = 40; ...; block_offset[4] = 80
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                // * TODO: it doesn't finish column_index yet!
                break;
            }
            
            // * if the current block `s_idx` is much larger than `range_end + block_size_m`, we don't extend it (in second condition, just save it). 
            // * instead, ignore [range_end, range_end + block_size_m), and use new one instead 
            if (s_idx > range_end + BLOCK_SIZE_M) {
                save_blocks(block_offset, range_start, range_end, BLOCK_SIZE_N, tmp_blk_cnt);
                range_start = s_idx - BLOCK_SIZE_M;
                range_end = s_idx;

            // * if false, it means current block is not enough the compute, we need to extend it
            } else if (s_idx > range_end) {
                range_end += BLOCK_SIZE_M;
            }
        }
    }

    // * both xxx_count is pointer that already move to correct location, simply deference it and assign value
    // * equivalent to `*block_count = tmp_blk_cnt`
    block_count[0] = tmp_blk_cnt;
    column_count[0] = tmp_col_cnt;
}

void convert_vertical_slash_indexes_64x64(
    const int* seqlens,           // [BATCH, ]
    const int* vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    const int* slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int* block_count,             // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* block_offset,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    int* column_count,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    int* column_index,            // [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    int BATCH_SIZE,
    int N_HEADS,
    int N_ROWS,
    int NNZ_V,
    int NNZ_S
) {
    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_N = 64;
    const int N_THREADS = 64;
    const dim3 dimBlock(N_THREADS);
    // * (N_ROWS + N_THREAD - 1) / N_THREADS -> ceil_div(#row, #threads) := assign #row to each thread. ie. row = 13, thread = 4, each thread handle 4 row, last_thread handle 1 row
    const dim3 dimGrid(N_HEADS, BATCH_SIZE, (N_ROWS + N_THREADS - 1) / N_THREADS);
    convert_vertical_slash_indexes_kernel<<<dimGrid, dimBlock>>>(
        seqlens, vertical_indexes, slash_indexes,
        block_count, block_offset, column_count, column_index,
        N_HEADS, N_ROWS, BLOCK_SIZE_M, BLOCK_SIZE_N, NNZ_V, NNZ_S
    );
}

std::vector<at::Tensor> convert_vertical_slash_indexes(
    torch::Tensor seqlens,           // [BATCH, ]
    torch::Tensor vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int context_size,
    int block_size_M,
    int block_size_N
) {
    assert(block_size_M == 64);
    assert(block_size_N == 64);

    cudaSetDevice(seqlens.get_device());

    int batch_size = slash_indexes.size(0);
    int num_heads = slash_indexes.size(1);

    // * nnz_xxx = seqlen * headdim
    int nnz_slash = slash_indexes.size(2);
    int nnz_vertical = vertical_indexes.size(2);
    int num_rows = (context_size + block_size_M - 1) / block_size_M;

    torch::Tensor block_count = torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
    torch::Tensor block_offset = torch::zeros({batch_size, num_heads, num_rows, nnz_slash}, seqlens.options());
    torch::Tensor column_count = torch::zeros({batch_size, num_heads, num_rows}, seqlens.options());
    torch::Tensor column_index = torch::zeros({batch_size, num_heads, num_rows, nnz_vertical}, seqlens.options());

    convert_vertical_slash_indexes_64x64(
        seqlens.data_ptr<int>(),
        vertical_indexes.data_ptr<int>(),
        slash_indexes.data_ptr<int>(),
        block_count.data_ptr<int>(),
        block_offset.data_ptr<int>(),
        column_count.data_ptr<int>(),
        column_index.data_ptr<int>(),
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        nnz_slash
    );

    return { block_count, block_offset, column_count, column_index };
}
