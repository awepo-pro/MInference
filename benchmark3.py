import torch
from typing import Optional, List
from dataclasses import dataclass
import types
import time

from minference.ops.block_sparse_flash_attention import (
    block_sparse_attention, 
    block_sparse_attention_with_kvcache
)

from vllm import _custom_ops as vllm_ops



def block_sparse_topk_vllm(q, k, v, head_id):
    # * q, k, v \in (batch=1, head=1, seqlen, head_size)
    def block_sparse_kernel(q, k, v, top_k=15):
        return block_sparse_attention(q, k, v, top_k)

    return block_sparse_kernel(q, k, v)


def block_sparse_topk_vllm_with_kvcache(
        q,                  # * (batch=1, #head=1, total_tokens, headdim) 
        k, 
        v,
        k_cache,            # * (#block, block_size, #head=1, headdim)
        v_cache,
        block_tables,        # * (#batch, max_num_block_per_seq)
        seq_lens,            # * (#batch)
    ) -> torch.Tensor:
    
    # * q \in (batch=1, head=1, total_tokens, head_size)
    bsz = q.shape[0]        # * bsz should be 1
    assert bsz == 1, f'bsz: {bsz} is not 1'

    # po_debug.debug_print(q)

    def block_sparse_kernel_with_kvcache(
            q, k, v, 
            k_cache, v_cache,
            # cu_seqlens_q, max_seqlne_q,
            # cu_seqlens_k, max_seqlen_k,
            block_tables,
            seq_lens: torch.Tensor, 
            top_k=15) -> torch.Tensor:
        return block_sparse_attention_with_kvcache(
            q, k, v,
            k_cache, v_cache,
            block_tables,
            top_k,
            seq_lens)

    return block_sparse_kernel_with_kvcache(
        q,                  # * (batch=1, #head=1, total_tokens, headdim) 
        k, 
        v,
        k_cache,            # * (#block, max_num_block_per_seq, #head=1, headdim)
        v_cache,
        # cu_seqlens_q,       # * (#batch + 1)
        # max_seqlen_q,
        # cu_seqlens_k, 
        # max_seqlen_k,
        block_tables,
        seq_lens)        # * (#batch, max_num_block_per_seq)


# ==========================================
# 1. Mock vLLM Structures & Constants
# ==========================================

@dataclass
class PrefillMetadata:
    block_tables: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None

@dataclass
class DecodeMetadata:
    block_tables: Optional[torch.Tensor] = None
    seq_lens_tensor: Optional[torch.Tensor] = None

@dataclass
class AttnMetadata:
    prefill_metadata: Optional[PrefillMetadata] = None
    decode_metadata: Optional[DecodeMetadata] = None
    slot_mapping: Optional[torch.Tensor] = None
    cross_slot_mapping: Optional[torch.Tensor] = None
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    # Additional fields to satisfy the get_num_prefill... helper
    is_prompt: bool = True 

class AttentionType:
    ENCODER = 0
    DECODER = 1
    ENCODER_DECODER = 2

def get_num_prefill_decode_query_kv_tokens(metadata, attn_type):
    """Mock helper to return token counts."""
    # In a real scenario, this logic is more complex, but for this test:
    return metadata.num_prefill_tokens, metadata.num_prefill_tokens, metadata.num_decode_tokens

def get_tensor_model_parallel_rank():
    return 0

# Monkey-patch torch.ops._C_cache_ops for KV Cache writing
# might not find _C_cache_ops in torch, since it might be built with vllm
# if not hasattr(torch.ops, "_C_cache_ops"):
#     torch.ops._C_cache_ops = types.SimpleNamespace()
#
# def mock_reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
#     """
#     Python implementation of the C++ kernel to write Q/K/V into the block cache.
#     slot_mapping contains linear indices. We assume flattened cache layout for simplicity 
#     or calculate block indices.
#     """
#     # Flatten caches for easier indexing: [num_blocks * block_size, num_kv_heads, head_size]
#     # Note: Real vLLM cache is [num_blocks, block_size, num_kv_heads, head_size]
#     # We will compute block indices from slot_mapping manually for the 5D tensor.
#
#     num_kv_heads = k_cache.shape[2]
#     head_size = k_cache.shape[3]
#     block_size = k_cache.shape[1]
#
#     # Iterate over tokens to cache
#     for i, slot in enumerate(slot_mapping):
#         # slot is a linear index. 
#         # block_idx = slot // block_size
#         # block_offset = slot % block_size
#         block_idx = slot.item() // block_size
#         block_offset = slot.item() % block_size
#
#         k_cache[block_idx, block_offset, :, :] = key[i]
#         v_cache[block_idx, block_offset, :, :] = value[i]
#
# torch.ops._C_cache_ops.reshape_and_cache_flash = mock_reshape_and_cache_flash


# ==========================================
# 2. Attention Layer with Your Code
# ==========================================

class MockAttentionLayer:
    def __init__(self):
        self.num_heads = 28
        self.num_kv_heads = 4
        self.head_size = 128
        self.scale = 1.0 / (self.head_size ** 0.5)
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = "auto"
        self.sliding_window = None
        self.alibi_slopes = None
        self.logits_soft_cap = None
        self.layer_idx = 0
        self._k_scale = 1.0
        self._v_scale = 1.0


    # --- YOUR PROVIDED CODE BELOW ---
    def forward_vllm_080(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        benchmark: bool = False
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size], #q_head = #heads = 14 (qwen-2)
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size], #kv_head = 2
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.

        def repeat_kv(hidden_states, n_rep):
            sqlen, num_head, head_dim = hidden_states.shape
            if n_rep == 1:
                return hidden_states
            hidden_states = hidden_states[:, :, None, :].expand(sqlen, num_head, n_rep, head_dim)
            return hidden_states.reshape(sqlen, num_head * n_rep, head_dim)

        def minference_prefill_func(
            q, k, v,
        ):
            # (seq_len, num_heads, head_size)
            if q.size(-2) != k.size(-2):
                k = repeat_kv(k, q.size(-2) // k.size(-2))
                v = repeat_kv(v, q.size(-2) // v.size(-2))

            assert k.shape == q.shape, f'{k.shape=} should be equivalent to {q.shape=}'

            output = torch.empty_like(q)
            head_idx_st = get_tensor_model_parallel_rank() * q.size(-2)
            for head in range(q.size(-2)):

                q_head = q[:, head, :].unsqueeze(1)
                k_head = k[:, head, :].unsqueeze(1)
                v_head = v[:, head, :].unsqueeze(1)

                # (1, seq_len, num_heads, head_size), batch_size = 1
                q_head = q_head[None, ...]
                k_head = k_head[None, ...]
                v_head = v_head[None, ...]

                q_head = q_head.transpose(1, 2)
                k_head = k_head.transpose(1, 2)
                v_head = v_head.transpose(1, 2)

                out = block_sparse_topk_vllm(q_head, k_head, v_head, head + head_idx_st)

                out = out.transpose(1, 2).squeeze(0).contiguous()
                output[:, head:head+1, :] = out
            return output
        
            
        def minference_prefill_kvcache_func(
                q: torch.Tensor,                # * (seqlen, #head, headdim), ragged batching
                k: torch.Tensor,                # * kv are cached in advanced, however, if len(k) == 1, we could do decode in this case
                v: torch.Tensor,
                k_cache: torch.Tensor,          # * (#block, max_num_block_per_seq=block_size, #head, headdim), block_size := 16 by default
                v_cache: torch.Tensor,  
                causal: bool,                   # * must be causal attention
                block_tables: torch.Tensor,     # * (#batch, max_num_block_per_seq)
        ) -> torch.Tensor:

            assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
            assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"

            q = q.contiguous() if q.stride(-1) != 1 else q
            k_n_rep = q.size(-2) // k.size(-2)
            v_n_rep = q.size(-2) // v.size(-2)

            assert k_n_rep == v_n_rep, f'{k_n_rep=} != {v_n_rep=}, kv should have same init heads in grouped multiple attention. remove if necessary'

            # (seq_len, num_heads, head_size)
            if q.size(-2) != k.size(-2):
                k = repeat_kv(k, k_n_rep)
                v = repeat_kv(v, v_n_rep)

            assert k.shape == q.shape, f'{k.shape=} != {q.shape=}'

            output = torch.empty_like(q)

            # --- FIX: Retrieve seq_lens correctly and ensure it is a tensor ---
            seq_lens = attn_metadata.prefill_metadata.seq_lens
            if not isinstance(seq_lens, torch.Tensor):
                seq_lens = torch.tensor(seq_lens, device=q.device, dtype=torch.int32)
            # ----------------------------------------------------------------

            for head in range(q.size(-2)):
                # * (seqlen, #head=1, headdim), unsqueeze(1) to make sure (#head=1) dimension doesn't disappear
                q_head = q[:, head, :].unsqueeze(1)
                k_head = k[:, head, :].unsqueeze(1)
                v_head = v[:, head, :].unsqueeze(1)


                # * (batch=1, seqlen, 1, headdim)
                q_head = q_head[None, ...]
                k_head = k_head[None, ...]
                v_head = v_head[None, ...]

                # * (1, 1, seqlen, headdim)
                q_head = q_head.transpose(1, 2)
                k_head = k_head.transpose(1, 2)
                v_head = v_head.transpose(1, 2)

                # * 1 head of kv cache, (#block, block_size, #head=1, headdim)
                k_cache_head = head // k_n_rep
                v_cache_head = head // v_n_rep
                assert k_cache_head < k_cache.size(2), f'{k_cache_head=} >= {k_cache.size(2)=}'
                assert v_cache_head < v_cache.size(2), f'{v_cache_head=} >= {v_cache.size(2)=}'

                k_head_cache = k_cache[:, :, k_cache_head, :].unsqueeze(2)
                v_head_cache = v_cache[:, :, v_cache_head, :].unsqueeze(2)

                out = block_sparse_topk_vllm_with_kvcache(
                    q_head, 
                    k_head,
                    v_head,
                    k_head_cache,               # * (#block, block_size, #head=1, headdim)
                    v_head_cache,
                    block_tables,
                    seq_lens)     # * FIXED: Passing tensor derived from attn_metadata.prefill_metadata

                # * transform into (n_ctx, n_heads, d_head)
                out = out.transpose(1, 2).squeeze(0).contiguous()

                output[:, head:head+1, :] = out

            return output
            

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        attn_type = self.attn_type
        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap
        fp8_attention = kv_cache_dtype.startswith("fp8")

        # assert kv_cache.shape[0] == 2, f'{kv_cache.shape}, first diemnsion must be 2'
        key_cache = kv_cache[0] if kv_cache is not None else None
        value_cache = kv_cache[1] if kv_cache is not None else None

        if key_cache is not None and value_cache is not None and kv_cache.numel() > 0:
            # assert False, f'{kv_cache.numel()=}'
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    torch.tensor(layer._k_scale),
                    torch.tensor(layer._v_scale),
                )

        num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens = get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        # * IMPORTANT: don't init after query=query[:num_prefill_query_tokens]
        output = torch.empty_like(query)
        assert output is not None

        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        # assert decode_query.shape[0] == num_decode_query_tokens

        # po_debug.debug_print(attn_metadata)
        used = 0

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            # * kv_cache.numel() != 0, prefill_meta.block_tables is not None
            if (kv_cache is None or kv_cache.numel() == 0 or prefill_meta.block_tables is None or prefill_meta.block_tables.numel() == 0):
                # po_debug.debug_print(query.shape)        # * (seqlen, #head=14, headdim=64), ie. "Hello my name is" -> (4, 14, 64)
                # po_debug.debug_print(key.shape)          # * (seqlen, #head=2, headdim=64)
                # po_debug.debug_print(value.shape)
                # po_debug.debug_print(num_prefill_query_tokens)   # * same as num_prefill_kv_tokens, ie. "Hello my name is" -> 4
                # po_debug.debug_print(num_prefill_kv_tokens)     
                # po_debug.debug_print(num_decode_query_tokens)    # * 0
                
                print("  [Logic Path] Entering Standard Prefill (minference_prefill_func)")
                print(f'{query.shape=}, {key.shape=}, {value.shape=}')
                torch.cuda.synchronize()
    
                start = time.time()

                with torch.no_grad():
                    out = minference_prefill_func(query, key, value)

                torch.cuda.synchronize()
                used = time.time() - start
                print(f'time: {used}')

                assert output[:num_prefill_query_tokens].shape == out.shape

                output[:num_prefill_query_tokens] = out
            else:
                print("  [Logic Path] Entering Prefix-Enabled Prefill (minference_prefill_kvcache_func)")
                print(f'{query.shape=}, {key.shape=}, {value.shape=}')

                # assert False
                # prefix-enabled attention, invoke by prefill chunk
                assert prefill_meta.seq_lens is not None
                    
                torch.cuda.synchronize()
    
                start = time.time()
                with torch.no_grad():
                    output[:num_prefill_query_tokens] = minference_prefill_kvcache_func(
                        query,
                        key,
                        value,
                        key_cache,
                        value_cache,
                        causal=True,
                        block_tables=prefill_meta.block_tables
                    )

                torch.cuda.synchronize()
                used = time.time() - start
                print(f'time: {used}')


                assert output.shape[0] == num_prefill_query_tokens, f'output size =({output.shape} not equivalent to {num_prefill_query_tokens}); actually not really if padded output, remove this line if necessary'

        # * it should be chunked prefill, that means decode and prefill mixed together
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            assert False
            pass # Skipped for this test case

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size), used if benchmark else None

# ==========================================
# 3. Test Harness: Prefix Attention
# ==========================================

import math

def test_minf_prefix_attention(prefix_len, total_len):
    warmup()
    print("=== Starting Prefix Attention Test Case ===")
    
    device = "cuda"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # 1. Initialize Layer and Cache
    layer = MockAttentionLayer()

    BLOCK_SIZE = 64

    num_blocks_needed = math.ceil(total_len / BLOCK_SIZE)
    print(f"  [Info] Blocks required: {num_blocks_needed}")

    kv_cache = torch.zeros(
        2, num_blocks_needed, BLOCK_SIZE, layer.num_kv_heads, layer.head_size,
        dtype=dtype, device=device
    )
    
    # --- STAGE 1: Standard Prefill ("Hello my name is") ---
    print("[Stage 1] Running Standard Prefill...")
    
    seq_len_1 = prefix_len
    
    q1 = torch.randn(seq_len_1, layer.num_heads * layer.head_size, device=device, dtype=dtype)
    k1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    v1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    
    # Slot mapping: indices [0, 1, 2, 3] in Block 0
    # Linear indices = block_idx * block_size + offset
    slot_mapping_1 = torch.arange(seq_len_1, device=device, dtype=torch.long) 
    
    meta_1 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=None,
            seq_lens=[prefix_len]
        ), # None triggers standard prefill
        slot_mapping=slot_mapping_1,
        num_prefill_tokens=seq_len_1
    )
    
    out1 = layer.forward_vllm_080(
        layer=layer, # Pass self as layer for property access
        query=q1, key=k1, value=v1, kv_cache=kv_cache,
        attn_metadata=meta_1
    )
    
    # Verify Cache was written
    # Check first token of key cache in block 0
    cached_k_0 = kv_cache[0, 0, 0, :, :].view(1, -1) # [1, 2*64]
    input_k_0 = k1[0].view(1, -1)
    if torch.allclose(cached_k_0, input_k_0):
        print("  [Check] KV Cache populated successfully in Stage 1.")
    else:
        print("  [Error] KV Cache mismatch in Stage 1!")

    # --- STAGE 2: Prefix-Enabled Prefill ("... Bob") ---
    print("[Stage 2] Running Prefix-Enabled Prefill...")
    
    seq_len_2 = total_len
    remains = seq_len_2 - prefix_len
    
    # Note: we don't torch.cat([q1, torch.randn()]), since q1 is shared prefix. Their kv could be found in kv cache
    q2 = torch.randn(remains, layer.num_heads * layer.head_size, device=device, dtype=dtype)
    k2 = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    v2 = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    
    # Slot mapping starts at index 4, length 10 -> [4, 5, ..., 13]
    slot_mapping_2 = torch.arange(prefix_len, total_len, device=device, dtype=torch.long)
    block_tables_2 = torch.arange(num_blocks_needed, dtype=torch.int32, device=device).unsqueeze(0)

    meta_2 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=block_tables_2, # Presence triggers prefix path
            seq_lens=[total_len] # Context length including prefix (List[int])
        ),
        slot_mapping=slot_mapping_2,
        num_prefill_tokens=remains
    )
    
    out2, used = layer.forward_vllm_080(
        layer=layer,
        query=q2, key=k2, value=v2, kv_cache=kv_cache,
        attn_metadata=meta_2,
        benchmark=True,
    )

    print("  [Success] Stage 2 completed.", end='\n\n')
    return used

def warmup():
    print("=== Starting Warm Up GPU ===")
    
    device = "cuda"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # 1. Initialize Layer and Cache
    layer = MockAttentionLayer()

    # --- STAGE 1: Standard Prefill ("Hello my name is") ---
    print("\n[Stage 1] Running Standard Prefill...")
    
    # Sequence length 4, fits in Block 0
    seq_len_1 = 100
    
    q1 = torch.randn(seq_len_1, layer.num_heads * layer.head_size, device=device, dtype=dtype)
    k1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    v1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    
    # Slot mapping: indices [0, 1, 2, 3] in Block 0
    # Linear indices = block_idx * block_size + offset
    slot_mapping_1 = torch.arange(seq_len_1, device=device, dtype=torch.long) 
    
    meta_1 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=None,
            seq_lens=[100]
        ), # None triggers standard prefill
        slot_mapping=slot_mapping_1,
        num_prefill_tokens=seq_len_1
    )
    
    out1 = layer.forward_vllm_080(
        layer=layer, # Pass self as layer for property access
        query=q1, key=k1, value=v1, kv_cache=None,      # kv_cache is empty
        attn_metadata=meta_1
    )

def test_minf(prefix_len, total_len):
    warmup()
    print("=== Starting MInference Attention Test Case ===")
    
    device = "cuda"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # 1. Initialize Layer and Cache
    layer = MockAttentionLayer()

    # --- STAGE 1: Standard Prefill ("Hello my name is") ---
    print("\n[Stage 1] Running Standard Prefill...")
    
    # Sequence length 4, fits in Block 0
    seq_len_1 = prefix_len
    
    q1 = torch.randn(seq_len_1, layer.num_heads * layer.head_size, device=device, dtype=dtype)
    k1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    v1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    
    # Slot mapping: indices [0, 1, 2, 3] in Block 0
    # Linear indices = block_idx * block_size + offset
    slot_mapping_1 = torch.arange(seq_len_1, device=device, dtype=torch.long) 
    
    meta_1 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=None,
            seq_lens=[prefix_len]
        ), # None triggers standard prefill
        slot_mapping=slot_mapping_1,
        num_prefill_tokens=seq_len_1
    )
    
    out1 = layer.forward_vllm_080(
        layer=layer, # Pass self as layer for property access
        query=q1, key=k1, value=v1, kv_cache=None,      # kv_cache is empty
        attn_metadata=meta_1,
    )
    
    # STAGE 2: New Suffix (10 tokens)
    seq_len_2 = total_len
    remains = seq_len_2 - prefix_len
   
    start = time.time()
    q2 = torch.cat([q1, torch.randn(remains, layer.num_heads * layer.head_size, device=device, dtype=dtype)])
    k2 = torch.cat([k1, torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)])
    v2 = torch.cat([v1, torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)])
    print(f'  [INFO]: generate and copy tensor, time used: {time.time() - start}')
    
    # Slot mapping starts at index 4, length 10 -> [4, 5, ..., 13]
    # slot_mapping_2 = torch.arange(prefix_len, total_len, device=device, dtype=torch.long)

    meta_2 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=None, # Presence triggers prefix path
            seq_lens=[total_len] # Context length including prefix (List[int])
        ),
        slot_mapping=None,
        num_prefill_tokens=seq_len_2
    )
    
    out2, used = layer.forward_vllm_080(
        layer=layer,
        query=q2, key=k2, value=v2, kv_cache=None,
        attn_metadata=meta_2,
        benchmark=True
    )

    print("  [Success] Stage 2 completed.")
    return used

# VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V1=0 ${PYTHON} benchmark3.py
if __name__ == "__main__":
    # test_minf_prefix_attention(2_000, 10_000)

    T = 10
    used = 0
    prefix = 200_000
    total = 500_000
    
    for _ in range(T):
        # used += test_minf(prefix, total)
        used += test_minf_prefix_attention(prefix, total)

    print(f'time: {used / T}')
