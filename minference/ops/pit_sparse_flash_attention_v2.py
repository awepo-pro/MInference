# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import triton
import triton.language as tl

from ..cuda import convert_vertical_slash_indexes

try:
    from sgl_kernel.sparse_flash_attn import sparse_attn_func
except:
    try:
        from vllm_flash_attn import sparse_attn_func
    except:
        print("To benefit from fast kernel implementations, we recommend installing SGLang or vllm.")
        sparse_attn_func = None

try:
    from sgl_kernel.sparse_flash_attn import (
        convert_vertical_slash_indexes as convert_vertical_slash_indexes_opt,
    )
except:
    try:
        from vllm._custom_ops import (
            convert_vertical_slash_indexes as convert_vertical_slash_indexes_opt,
        )
    except:
        convert_vertical_slash_indexes_opt = None

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
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,         # * stride of (b, h, seqlen, headdim)
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,                                        # * Z := batch_size; H := #head; N_CTX := seqlen
    NUM_ROWS, NNZ_S, NNZ_V,                             # * NUM_ROWS := ceil_div(seqlen, block_size_M);
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,                         # * headdim
    dtype: tl.constexpr,
):
    # * program_id(0) := blockIdx.x -> index of query block (ceil_div(q_seqlen, block_size_M)); program_id(1) := blockIdx.y -> b * h
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # * off_hz := batch_size * #head -> off_hz // H := Z, H := #head, Z := batch_size
    # * seqlens \in (batch, ); each batch has 1 sequence only. we say that batching means several number of sequences (batches) pass forward at once.
    # * seqlen := that len of that sequence;
    seqlen = tl.load(seqlens + off_hz // H) 
    # * if means if exceed seqlen (padding) do nothing
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    # * start_m := index of query block -> start * BLOCK_M := start point of that query block
    # * tl.arange(0, BLOCK_M) := [0, 1, 2, ..., BLOCK_M - 1]
    # * offs_m := list of indices record global index of query, \in (BLOCK_M)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # * offs_n := list of indices record relative index of kv
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    """Assume head dim = 64
    stride_qk = 1                   # features are adjacent
    stride_qm = 64                  # skip 64 elements to next token
    stride_qh = 64 * 512            # skip 64*512 elements to next head
    stride_qz = 64 * 512 * 4        # skip 64*512*4 elements to next batch
    """

    # * off_hz // H := batch_idx; off_hz % H := head_idx in (off_hz // H)-th batch
    # * xx_offset := offset of batch_size and #head
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    # * offs_m[:, None] \in (BLOCK_M, 1) 
    q_ptrs = Q      + qo_offset     + offs_m[:, None] * stride_qm   + offs_d[None, :] * stride_qk
    k_ptrs = K      + kv_offset                                     + offs_d[:, None] * stride_kk
    v_ptrs = V      + kv_offset                                     + offs_d[None, :] * stride_vk
    o_ptrs = Out    + qo_offset     + offs_m[:, None] * stride_om   + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    # * block_offset \in [BATCH, N_HEADS, NUM_ROWS=cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    # * (off_hz * NUM_ROWS + start_m) * NNZ_S := (b * h * #rows * NNZ_S) + (start_m * NNZ_S); start_m * NNZ_S := which block (corresponding to NUM_ROWS)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    # * column_index \in [BATCH, N_HEADS, NUM_ROWS=cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # * 1.442 := log_2(e)
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    # * [[true, true, ..., false, false]], remove padding
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        # * num_blks := number of blocks used. num_blks <= NNZ_S
        # * start_n := exact starting index, note that it is a scalar
        start_n = tl.load(blks_ptr + block_index)

        # * convert into indices. cols[0] = starting; cols[-1] = ending
        cols = start_n + offs_n
        
        # * mask to avoid OOB
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        """ causal mask -> lower-triangular matrix
        Col 0,Col 1,Col 2,Col 3 
        Row 0,T,F,F,F
        Row 1,T,T,F,F
        Row 2,T,T,T,F
        Row 3,T,T,T,T

        col := kv (exact kv location)
        row := query (offs_m := exact query location)
        """
        causal_mask = cols[None, :] <= offs_m[:, None]
        # * masking 
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        # * exp(m_i - m_i^{new}) * acc_old
        acc *= acc_scale[:, None]
        # * acc_old + acc_new
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
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
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD], batch := batch_size -> no; n_heads := no. of heads in a batch
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    # * headdim constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    # * output size
    o = torch.zeros_like(q)

    # * (ceil_div(seqlen, block_size_M), b * h, 1)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    if convert_vertical_slash_indexes_opt is not None:
        return vertical_slash_sparse_attention_wo_pad(query, key, value, v_idx, s_idx)
    batch_size, num_heads, context_size, head_dim = query.shape

    # * pad N_CTX dimension
    pad = (block_size_M - context_size) & (block_size_M - 1)
    # * .paed(input, [(D_HEAD1, D_HEDAD2), (N_CTX1, N_CTX2), (N_HEADS1, N_HEADS2), ...])
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    # * pad D_HEAD dimension
    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    # * sort to allow sequential access
    # * v_idx reshape from (b, h, seqlen, headim) to (b, h, seqlen * headdim). 
    # * reshape(batch_size, num_heads, -1) mean I know first 2 dimension is batch_size, num_heads, the third dimension is wlidcard
    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5

    # * block_size_M = block_size_N = 64 (must)
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
    )

    if sparse_attn_func is not None:
        out = sparse_attn_func(
            # * (b, seqlen, head_len, headdim)
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            block_count, block_offset, column_count, column_index,
            return_softmax_lse=False,
            causal=True,
        ).transpose(1, 2).contiguous()
    else:
        out = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, block_size_M, block_size_N,
        )

    return out[..., :context_size, :head_dim]

def vertical_slash_sparse_attention_wo_pad(query, key, value, v_idx, s_idx, block_size_M: int = 64, block_size_N: int = 64):
    batch_size, num_heads, context_size, head_dim = query.shape
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    
    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    
    block_count, block_offset, column_count, column_index = (
        convert_vertical_slash_indexes_opt(
            seqlens,
            seqlens,
            v_idx.to(torch.int32),
            s_idx.to(torch.int32),
            context_size,
            block_size_M,
            block_size_N,
            causal=True,
        )
    )
    out = sparse_attn_func(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=True,
        return_softmax_lse=False,
    )
    return out.transpose(1, 2).contiguous()
