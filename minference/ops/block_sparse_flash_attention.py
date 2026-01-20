# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import torch
import triton
import triton.language as tl

# from flash_attn import flash_attn_varlen_func
# import pycuda.autoprimaryctx
# from pycuda.compiler import SourceModule

import po_debug


# * https://claude.ai/share/5645c803-86b6-4458-8d34-c60422975833
def _build_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    po_debug.debug_print(key.shape)
    po_debug.debug_print(query.shape)

    # * query.reshape := (b, n, seqlen, headdim) -> (b, n, seqlen // block_size_M, block_size_M, headdim)
    # * query.reshape.mean(dim=-2) := (b, n, seqlen // block_size_M, block_size_M, headdim) -> (b, n, seqlen // block_size_M, headdim)
    # * it means compress seqlen into blocks by averaging them 
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)

    # * arange(end=query_pool.shape[-2]) := arange(seqlen // block_size_M) -> [0, 1, 2, ..., seqlen // block_size_M)
    # * arange * block_size_M := starting position of each query block
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N

    # * (b, n, q_seqlen // block_size_M, k_seqlen // block_size_N)
    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    # * build 4D, arrange_M \in (b=1, h=1, m, 1); arange_N \in (b=1, h=1, 1, n). build a mask in last 2 dimension. should be a upper-triangular matrix
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)

    po_debug.debug_print(p_pool.shape)

    # * top_k cannot exceed p_pool[-1] dimension
    top_k = min(top_k, context_size // block_size_N)
    
    # * find topk row by row,
    # * topk.indices \in (b, h, m, topk), topk := scalar
    # * indices.sort return (values, indices), since we sort the indices, so values = indices
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )
@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V,                            # * (b, h, seqlen, headdim)
    seqlens, sm_scale,
    block_index,                        # * (b, h, ceil_div(seqlen, block_size_M), topk=MAX_BLOCKS_PER_ROW)
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,                        # * Z=q.shape[0], H=q.shape[1], N_CTX=q.shape[2]
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,       # * NUM_ROWS=BLOCK_SIZE_M; MAX_BLOCKS_PER_ROW := min(topk, seqlen // block_size_N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    # * ceil_div(seqlen, block_size_M), index of starting block
    start_m = tl.program_id(0)
    # * b \times h
    off_hz = tl.program_id(1)

    # * seqlen of corresponding batch
    seqlen = tl.load(seqlens + off_hz // H)
    # * do nothing if padding
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    # * we treat QKV as \in (b, h, seqlen // block_size_M, headdim)

    # * start_m * block_M := starting position of current block (among seqlen // block_size_M)
    # *     - 4 blocks in 1 head (seqlen = 128) -> 1st: [0, 32), 2nd: [32, 64), 3th: [64, 96), 4th: [96, 128)
    # *     - now in 3th -> start * block_M = 64; +tl.arange(0, block_M) := [32, 64)

    # * converted into list with `tl.arange` to get the exact location of each elements
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # * offset of batch and head
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    # * why uses stride to compute q_ptrs and shape to compute blocks_ptr?
    # *     - Q might not contiguous tensor, stride is generalized method 
    # *     - blocks_ptr is contiguous, might use size of stride to compute

    # tl.device_print(H)
    # tl.device_print(off_hz // H)
    # tl.device_print(off_hz % H)
    # tl.device_print(stride_qz)
    # tl.device_print(stride_qh)
    # tl.device_print(stride_qm)
    # tl.device_print(stride_qk)
    # tl.device_print(qo_offset)
    

    # * start_point + batch_head_offset + seqlen_offset + headdim_offset
    q_ptrs = Q      + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K      + kv_offset                               + offs_d[:, None] * stride_kk
    v_ptrs = V      + kv_offset                               + offs_d[None, :] * stride_vk
    o_ptrs = Out    + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # * NUM_ROWS := no. of rows in each block
    # * off_hz * NUM_ROWS := move to the current head; start_m := determine current block
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    # * q_ptrs is a list, so load an array
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW], MAX_BLOCKS_PER_ROW := min(topk, seqlen // block_size_N)
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # * ================================ dbug print ========================================
    # po_debug.debug_print(q.shape)
    # off_hz = grid[1]
    # H = q.shape[1]
    # po_debug.debug_print(off_hz)
    # po_debug.debug_print(off_hz // H)
    # po_debug.debug_print(off_hz % H)

    # * ====================================================================================
    
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o

@triton.jit
def _triton_block_sparse_attn_fwd_kernel_with_kvcache(
    Q,                            # * (b=1, h=1, seqlen, headdim)
    k_cache,                        # * (#block, block_size=256, #kv_head=1, headdim)
    v_cache,
    q_seqlen,                       # * scalar
    k_seqlen,
    block_tables,                 # * (#batch=1, max_num_block_per_seq=max_seq / block), max_seq := max model len, block := 16 (by default)
    bt_batch,                     # * #batch                in block_tables
    bt_num_block,                     # * max_num_block_per_seq in block_tables
    stride_bt_a,                    # * block_tables.stride(0)
    stride_bt_b,                    # * block_tables.stride(1)
    sm_scale,
    block_index,                        # * (b, h, NUM_ROWS=ceil_div(seqlen, block_size_M), topk=MAX_BLOCKS_PER_ROW)
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kblock, stride_kblock_size, stride_num_khead, stride_k_headdim,     
    stride_vblock, stride_vblock_size, stride_num_vhead, stride_v_headdim,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,                        # * Z, H, N_CTX := q.shape[0, 1, 2]
    NUM_ROWS: tl.constexpr, MAX_BLOCKS_PRE_ROW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # * ceil_div(seqlen, block_size_M), index of starting block
    start_m = tl.program_id(0)
    # * b \times h
    off_hz = tl.program_id(1)

    off_a = tl.arange(0, BLOCK_DMODEL)[:, None] * stride_k_headdim
    off_b = tl.arange(0, BLOCK_N)[None, :] * stride_kblock_size
    k_tmp = tl.load(k_cache + off_a + off_b)
    tl.device_print('k-tmp: ', k_tmp)

    # assert off_hz == 0, f'{off_hz=} != 0'

    # * seqlen of corresponding batch
    seqlen = q_seqlen
    # * do nothing if padding
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    # * we treat QKV as \in (b, h, seqlen // block_size_M, headdim)

    # * start_m * block_M := starting position of current block (among seqlen // block_size_M)
    # *     - 4 blocks in 1 head (seqlen = 128) -> 1st: [0, 32), 2nd: [32, 64), 3th: [64, 96), 4th: [96, 128)
    # *     - now in 3th -> start * block_M = 64; +tl.arange(0, block_M) := [32, 64)

    # * converted into list with `tl.arange` to get the exact location    + offs_d[:, None] * stride_of each elements
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # * batch_id := 0; head_id := 0
    batch_id = off_hz // H
    head_id = off_hz // Z

    # * offset of batch and head, 
    # * assert qo_offset == 0
    qo_offset = batch_id * stride_qz + (off_hz % H) * stride_qh
    # tl.device_print('off_hz % H: ', off_hz % H)             # * = 0

    # * why uses stride to compute q_ptrs and shape to compute blocks_ptr?
    # *     - Q might not contiguous tensor, stride is generalized method 
    # *     - blocks_ptr is contiguous, might use size of stride to compute

    # * start_point + batch_head_offset + block_offset + headdim_offset
    o_ptrs = Out    + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    q_ptrs = Q      + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # tl.device_print('qo_offset: ', qo_offset)                                       # * qo_offset = 0
    # tl.device_print('q_ptrs: ', qo_offset + offs_m[:, None])                        # * [0, block_M)


    # * starting_point + #head + headdim offset 
    # * off_d[:, None] \in (BLOCK_DMODEL, 1)
    # * k_base_ptrs \in (BLOCK_DMODEL, 1); 
    k_base_ptrs = k_cache + head_id * stride_num_khead + offs_d[:, None] * stride_k_headdim
    # tl.device_print('head_id: ', head_id)       # * head_id = 0
    # tl.device_print('head_id * stride_num_khead: ', head_id * stride_num_khead)   # * = 0
    # * v_base_ptrs \in (1, BLOCK_DMODEL)
    v_base_ptrs = v_cache + head_id * stride_num_vhead + offs_d[None, :] * stride_v_headdim

    # * NUM_ROWS := no. of rows in each block
    # * off_hz * NUM_ROWS := move to the current head; start_m := determine current block
    # * assert (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW == start_m * MAX_BLOCKS_PRE_ROW
    blocks_ptr = block_index + start_m * MAX_BLOCKS_PRE_ROW
    # tl.device_print('new blocks ptr: ', start_n * MAX_BLOCKS_PRE_ROW)     # * starting position of block index (block_index.shape[0] := num of query blocks)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    # * q_ptrs is a list, so load an array
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < q_seqlen
    # tl.device_print('m_mask: ', m_mask)
    # tl.device_print('offs_m: ', offs_m[:, None])
    # tl.device_print('q_seqlen: ', q_seqlen)
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N

        bt_index = start_n // BLOCK_SIZE     # * bt_index := block table index inside block tables; BLOCK_SIZE := k_cache.shape[1]
        bt_block_index = start_n % BLOCK_SIZE    # * bt_block_index := exact block inside that block table

        physical_idx = block_tables \
                        + 0 * stride_bt_a \
                        + bt_index * stride_bt_b

        physical_index = tl.load(physical_idx)
        # tl.device_print('physical_index: ', physical_index)

        cols = start_n + offs_n

        # * #block, block_size
        # * k_base_ptrs \in (BLOCK_DMODEL, 1) \plus (1, BLOCK_N) -> (BLOCK_DEMOEL, BLOCK_N)
        k_ptrs = k_base_ptrs \
                + physical_index * stride_kblock \
                + (bt_block_index + offs_n)[None, :] * stride_kblock_size
        
        # tl.device_print('physical_index * stride_kblock: ', physical_index * stride_kblock) # * = 0
        # tl.device_print('bt_block_index: ', bt_block_index)   # * = 0
        # tl.device_print('bt_block_index + offs_n: ', bt_block_index + offs_n) # * 0, 1, 2, ..., 31
        # tl.device_print('(bt_block_index + offs_n)[None, :] * stride_kblock_size: ', (bt_block_index + offs_n)[None, :] * stride_kblock_size) # * 0, 64, 128, ..., 1920, 1984
        
        v_ptrs = v_base_ptrs \
                + physical_index * stride_vblock \
                + (bt_block_index + offs_n)[:, None] * stride_vblock_size

        # -- load k, v --
        k = tl.load(k_ptrs)

        # tl.device_print('k: ', k.shape_tuple())
        # tl.device_print('offs_d[:, None] * stride_k_headdim: ', offs_d[:, None] * stride_k_headdim)
        # tl.device_print('actual position: ', head_id * stride_num_khead + offs_d[:, None] * stride_k_headdim + physical_index * stride_kblock + (bt_block_index + offs_n)[None, :] * stride_kblock_size)
        # tl.device_print('actual k: ', k)
        # idx = head_id * stride_num_khead + offs_d[:, None] * stride_k_headdim + physical_index * stride_kblock + (bt_block_index + offs_n)[None, :] * stride_kblock_size
        # tl.device_print('idx: ', idx)
        # tl.device_print('actual k[0], ', k)
    
        v = tl.load(v_ptrs)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:

        # tl.device_print('k_seqlen: ', k_seqlen)
        # tl.device_print('q_seqlen: ', q_seqlen)
        q_preced_len = k_seqlen - q_seqlen
        q_absolute = q_preced_len + start_m * BLOCK_M
        abs_offs_m = q_absolute + tl.arange(0, BLOCK_M)
        
        causal_mask = cols[None, :] <= abs_offs_m[:, None]
        # tl.device_print('cols: ', cols)
        # tl.device_print('abs_offs_m: ', abs_offs_m)
        # tl.device_print('causal_mask: ', causal_mask)
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention_with_kvcache(
    q,                 # * [BATCH=1, N_HEADS=1, N_CTX, D_HEAD]
    k_cache,           # * (#block, block_size=256, #kv_head=2, headdim)
    v_cache,
    q_seqlen,          # * scalar
    k_seqlen,
    block_tables,      # * (#batch, max_num_block_per_seq)
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW], MAX_BLOCKS_PER_ROW := min(topk, seqlen // block_size_N)
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    headdim = q.shape[-1]
    assert headdim in {16, 32, 64, 128}
    assert q.shape[0] == 1, f'batch size should be 1, but ({q.shape[0]})'

    # * ================================================= dbug =====================================================
    # po_debug.debug_print(q.shape)                # * (1, 1, 512, 64)
    # po_debug.debug_print(q_seqlen)
    # po_debug.debug_print(k_seqlen)
    # po_debug.debug_print(block_tables.shape)
    po_debug.debug_print(block_index)
    po_debug.debug_print(block_index.shape)
    # po_debug.debug_print(k_cache.stride(1))
    po_debug.debug_print(k_cache.shape)
    po_debug.debug_print(k_cache.stride())
    po_debug.debug_print(k_cache[0][:10])
    po_debug.debug_print(k_cache.flatten()[:200])
    po_debug.debug_print(k_cache.flatten()[0].to(torch.float16))
    po_debug.debug_print(k_cache.flatten()[64].to(torch.float32))    
    po_debug.debug_print(k_cache.flatten()[128].to(torch.float32))    

    # po_debug.debug_print(k_cache[0][:q_seqlen][0])
    
    # * ============================================================================================================

    assert q.shape[0] * q.shape[1] == 1, f'{q.shape[0]=}, {q.shape[1]=}, {(q.shape[0] * q.shape[1])=} != 1'

    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    BLOCK_SIZE = k_cache.shape[1]


    # * grid := (#block_M, 1)
    _triton_block_sparse_attn_fwd_kernel_with_kvcache[grid](
        q, 
        k_cache, 
        v_cache, 
        q_seqlen,
        k_seqlen,
        block_tables,
        block_tables.shape[0], block_tables.shape[1],
        block_tables.stride(0), block_tables.stride(1),
        sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache.stride(0), k_cache.stride(1) // 2, k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z=q.shape[0], H=q.shape[1], N_CTX=q.shape[2],
        NUM_ROWS=block_index.shape[-2], MAX_BLOCKS_PRE_ROW=block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=headdim,
        dtype=dtype,
        # num_warps=4, num_stages=2,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return o


def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64, # might change to 16 (follow vllm block size)
    block_size_N: int = 64, # might change to 16 (follow vllm block size)
):
    batch_size, num_heads, context_size, head_dim = query.shape

    assert batch_size == 1, f'{batch_size=} != 1'
    assert num_heads == 1, f'{num_heads=} != 1'
    
    pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)

    sm_scale = head_dim ** -0.5
    block_index = _build_block_index(query, key, top_k, block_size_N, block_size_N)

    with open('output2.txt', 'a') as output:
        po_debug.debug_print(block_index, comment="normal", out=output)
    
    out = _triton_block_sparse_attention(
        query, key, value, 
        seqlens,
        block_index, 
        sm_scale,
        block_size_M, block_size_N)
    return out[..., :context_size, :]


def get_full_key_from_cache(k_cache, block_tables, seqlen):
    """
    Reconstruct full key tensor from paged k_cache.
    
    Args:
        k_cache: (#block, block_size, #kv_head, headdim)
        block_tables: (#batch=1, max_block_per_seq)
        seqlen: actual sequence length
        
    Returns:
        key: (#batch, #kv_head, seqlen, headdim)
    """

    # po_debug.debug_print(k_cache.shape)
    
    batch_size = block_tables.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # Calculate number of blocks needed for seqlen
    num_blocks_needed = (seqlen + block_size - 1) // block_size if seqlen != -1 else block_tables.shape[1]
    
    # Gather blocks for each sequence in batch
    # block_tables: (#batch, max_block_per_seq)
    # We take only the first num_blocks_needed blocks
    block_indices = block_tables[:, :num_blocks_needed]  # (#batch, num_blocks_needed)
    
    # Gather the blocks from k_cache
    # k_cache[block_indices] would give us (#batch, num_blocks_needed, block_size, #kv_head, headdim)
    gathered_blocks = k_cache[block_indices]  # (#batch, num_blocks_needed, block_size, #kv_head, headdim)
    
    # Reshape to merge blocks into sequence dimension
    # (#batch, num_blocks_needed * block_size, #kv_head, headdim)
    full_key = gathered_blocks.reshape(batch_size, num_blocks_needed * block_size, num_kv_heads, head_dim)
    
    # Trim to actual sequence length
    full_key = full_key[:, :seqlen, :, :]  # (#batch, seqlen, #kv_head, headdim)
    
    # Transpose to match desired output shape: (#batch, #kv_head, seqlen, headdim)
    full_key = full_key.transpose(1, 2)  # (#batch, #kv_head, seqlen, headdim)
    
    return full_key

# Usage:
# key = get_full_key_from_cache(k_cache, block_tables, seqlen)

def _build_block_index_with_kvcache(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k_cache: torch.Tensor,       # * (#block, block_size, #kv_head=1, headdim)
    block_tables: torch.Tensor,     # * (#batch=1, max_block_per_seq)
    top_k: int,
    k_seqlen,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape

    # po_debug.debug_print(k_seqlen)

    key = get_full_key_from_cache(k_cache, block_tables, k_seqlen)
    po_debug.debug_print(key.shape)
    po_debug.debug_print(query.shape)

    # * query.reshape := (b, n, seqlen, headdim) -> (b, n, seqlen // block_size_M, block_size_M, headdim)
    # * query.reshape.mean(dim=-2) := (b, n, seqlen // block_size_M, block_size_M, headdim) -> (b, n, seqlen // block_size_M, headdim)
    # * it means compress seqlen into blocks by averaging them 
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)

    # * arange(end=query_pool.shape[-2]) := arange(seqlen // block_size_M) -> [0, 1, 2, ..., seqlen // block_size_M)
    # * arange * block_size_M := starting position of each query block
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N

    # po_debug.debug_print(query_pool.shape)
    # po_debug.debug_print(key_pool.shape)

    # * (b, n, q_seqlen // block_size_M, k_seqlen // block_size_N)
    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    # * build 4D, arrange_M \in (b=1, h=1, m, 1); arange_N \in (b=1, h=1, 1, n). build a mask in last 2 dimension. should be a upper-triangular matrix
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)

    # * top_k cannot exceed p_pool[-1] dimension
    top_k = min(top_k, k_seqlen // block_size_N)

    # po_debug.debug_print(k_seqlen)
    # po_debug.debug_print(block_size_N)
    # po_debug.debug_print(k_seqlen // block_size_N)
    # po_debug.debug_print(top_k)
    po_debug.debug_print(p_pool.shape)
    
    # * find topk row by row,
    # * topk.indices \in (b, h, m, topk), topk := scalar
    # * indices.sort return (values, indices), since we sort the indices, so values = indices
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values


def block_sparse_attention_with_kvcache(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k_cache: torch.Tensor,                  # * (#block, block_size, #kv_head=1, headdim)
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,             # * (#batch, block_size), #batch == 1
    top_k: int,
    k_seqlen: torch.Tensor,
    block_size_M: int = 64, # might change to 16 (follow vllm block size)
    block_size_N: int = 64, # might change to 16 (follow vllm block size)
):
    
    # po_debug.debug_print(query.shape)
    # po_debug.debug_print(key.shape)
    
    batch_size, num_heads, context_size, head_dim = query.shape

    assert batch_size == 1, f'{batch_size=} != 1'
    assert num_heads == 1, f'{num_heads=} != 1'
    assert block_tables.shape[0] == 1, f'{block_tables.shape=}, where shape[0] != 1'
    
    q_seqlen = query.shape[-2]
    q_pad = block_size_M - (query.shape[2] & (block_size_M - 1))

    if q_pad != block_size_M:
        # po_debug.debug_print(block_size_M)
        query = torch.nn.functional.pad(query, [0, 0, 0, q_pad, 0, 0, 0, 0])

    kv_pad = block_size_N - (key.shape[2] & (block_size_N - 1))
    if kv_pad != block_size_N:
        # po_debug.debug_print(block_size_N)
        key = torch.nn.functional.pad(key, [0, 0, 0, q_pad, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, 0, 0, q_pad, 0, 0, 0, 0])

    # po_debug.debug_print(q_pad)
    # po_debug.debug_print(query.shape)
    # po_debug.debug_print(key.shape)

    all_kv_pad = k_seqlen[0] + int(block_size_N - (k_seqlen[0] & (block_size_N - 1)))
    # po_debug.debug_print(k_seqlen[0])
    # po_debug.debug_print(all_kv_pad)
    # po_debug.debug_print(k_seqlen[0] % block_size_N == int(block_size_N - (k_seqlen[0] & (block_size_N - 1))))


    sm_scale = head_dim ** -0.5
    block_index = _build_block_index_with_kvcache(
        query, k_cache, block_tables,
        top_k, 
        all_kv_pad,
        block_size_N, block_size_N)
    
    
    
    with open('output.txt', 'a') as output:
        po_debug.debug_print(block_index, comment="kv cache", out=output)
        
    
    out = _triton_block_sparse_attention_with_kvcache(
        query,
        k_cache, 
        v_cache,
        q_seqlen,
        k_seqlen[0],
        block_tables,
        block_index, 
        sm_scale,
        block_size_M, block_size_N)
    
    return out[..., :context_size, :]
