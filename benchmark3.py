import torch
from typing import Optional, List
from dataclasses import dataclass
import types
import time
import math

# Try importing, mock if not available for standalone testing purposes
try:
    from minference.ops.block_sparse_flash_attention import (
        block_sparse_attention, 
        block_sparse_attention_with_kvcache
    )
except ImportError as e:
    print(f'{e=}')
    # Mock for testing if minference is not installed
    def block_sparse_attention(q, k, v, top_k):
        return torch.zeros_like(q)
    def block_sparse_attention_with_kvcache(q, k, v, k_c, v_c, bt, top_k, sl):
        return torch.zeros_like(q)


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
if not hasattr(torch.ops, "_C_cache_ops"):
    torch.ops._C_cache_ops = types.SimpleNamespace()

def mock_reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
    """
    Python implementation of the C++ kernel to write Q/K/V into the block cache.
    slot_mapping contains linear indices. We assume flattened cache layout for simplicity 
    or calculate block indices.
    """
    # Flatten caches for easier indexing: [num_blocks * block_size, num_kv_heads, head_size]
    # Note: Real vLLM cache is [num_blocks, block_size, num_kv_heads, head_size]
    # We will compute block indices from slot_mapping manually for the 5D tensor.
    
    num_kv_heads = k_cache.shape[2]
    head_size = k_cache.shape[3]
    block_size = k_cache.shape[1]
    
    # Iterate over tokens to cache
    for i, slot in enumerate(slot_mapping):
        # slot is a linear index. 
        # block_idx = slot // block_size
        # block_offset = slot % block_size
        block_idx = slot.item() // block_size
        block_offset = slot.item() % block_size
        
        k_cache[block_idx, block_offset, :, :] = key[i]
        v_cache[block_idx, block_offset, :, :] = value[i]

torch.ops._C_cache_ops.reshape_and_cache_flash = mock_reshape_and_cache_flash


# ==========================================
# 2. Attention Layer with Your Code
# ==========================================

class MockAttentionLayer:
    def __init__(self):
        # Qwen-2 0.5B style config
        self.num_heads = 14
        self.num_kv_heads = 2
        self.head_size = 64
        self.scale = 1.0 / (self.head_size ** 0.5)
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = "float16"
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

        assert kv_cache.shape[0] == 2, f'{kv_cache.shape}, first diemnsion must be 2'
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        if kv_cache.numel() > 0:
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
                    layer._k_scale,
                    layer._v_scale,
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
        assert decode_query.shape[0] == num_decode_query_tokens

        # po_debug.debug_print(attn_metadata)

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            # * kv_cache.numel() != 0, prefill_meta.block_tables is not None
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None or prefill_meta.block_tables.numel() == 0):
                # po_debug.debug_print(query.shape)        # * (seqlen, #head=14, headdim=64), ie. "Hello my name is" -> (4, 14, 64)
                # po_debug.debug_print(key.shape)          # * (seqlen, #head=2, headdim=64)
                # po_debug.debug_print(value.shape)
                # po_debug.debug_print(num_prefill_query_tokens)   # * same as num_prefill_kv_tokens, ie. "Hello my name is" -> 4
                # po_debug.debug_print(num_prefill_kv_tokens)     
                # po_debug.debug_print(num_decode_query_tokens)    # * 0
                
                print("  [Logic Path] Entering Standard Prefill (minference_prefill_func)")
                out = minference_prefill_func(query, key, value)
                assert output[:num_prefill_query_tokens].shape == out.shape

                output[:num_prefill_query_tokens] = out
            else:
                print("  [Logic Path] Entering Prefix-Enabled Prefill (minference_prefill_kvcache_func)")
                # prefix-enabled attention, invoke by prefill chunk
                assert prefill_meta.seq_lens is not None
                    
                output[:num_prefill_query_tokens] = minference_prefill_kvcache_func(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    causal=True,
                    block_tables=prefill_meta.block_tables
                )

                assert output.shape[0] == num_prefill_query_tokens, f'output size =({output.shape} not equivalent to {num_prefill_query_tokens}); actually not really if padded output, remove this line if necessary'

        # * it should be chunked prefill, that means decode and prefill mixed together
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            assert False
            pass # Skipped for this test case

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

# ==========================================
# 3. Test Harness: Prefix Attention
# ==========================================

def test_long_prefix_attention(target_seq_len=1_000_000):
    print(f"=== Starting Long Sequence Test: {target_seq_len} tokens ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    layer = MockAttentionLayer()
    BLOCK_SIZE = 16
    
    # 1. Calculate required blocks
    # Formula: ceil(total_len / block_size)
    num_blocks_needed = math.ceil(target_seq_len / BLOCK_SIZE)
    print(f"  [Info] Blocks required: {num_blocks_needed}")

    # 2. Initialize a large enough KV Cache
    # In a real system, this is pre-allocated by the BlockManager
    kv_cache = torch.zeros(
        2, num_blocks_needed, BLOCK_SIZE, layer.num_kv_heads, layer.head_size,
        dtype=dtype, device=device
    )
    
    # 3. Define the prefix and the new query
    # Let's assume we have a massive prefix and we are adding 1 new token (Prefill of the next chunk)
    prefix_len = target_seq_len - 1
    new_tokens_len = 1 
    
    # 4. Generate block_tables_2
    # This is an array of physical indices into the 2nd dimension of kv_cache
    # For this test, we assume a simple linear mapping: Block 0, Block 1, Block 2...
    block_tables_2 = torch.arange(num_blocks_needed, dtype=torch.int32, device=device).unsqueeze(0) 
    # Shape: (batch_size=1, num_blocks_needed)

    # 5. Generate slot_mapping
    # Every token in the 'new' query needs a slot index
    # Formula: block_idx * block_size + block_offset
    # For a token at position 'i', slot = i
    slot_mapping_2 = torch.arange(prefix_len, target_seq_len, device=device, dtype=torch.long)

    # 6. Define Metadata
    meta_2 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=block_tables_2, 
            seq_lens=[target_seq_len] # Total length including the prefix
        ),
        slot_mapping=slot_mapping_2,
        num_prefill_tokens=new_tokens_len
    )

    # Dummy Tensors for the new query
    q2 = torch.randn(new_tokens_len, layer.num_heads * layer.head_size, device=device, dtype=dtype)
    k2 = torch.randn(new_tokens_len, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)
    v2 = torch.randn(new_tokens_len, layer.num_kv_heads * layer.head_size, device=device, dtype=dtype)

    print(f"  [Running] Executing forward for {target_seq_len} context...")
    
    start = time.time()
    with torch.no_grad():
        out = layer.forward_vllm_080(
            layer=layer,
            query=q2, key=k2, value=v2, kv_cache=kv_cache,
            attn_metadata=meta_2
        )
    
    if device == "cuda": torch.cuda.synchronize()
    print(f"  [Success] Time taken: {time.time() - start:.4f}s")

if __name__ == "__main__":
    # Note: 1M tokens with fp16/head_size 64/2 KV heads is ~256MB per layer.
    # Adjust target_seq_len based on your available VRAM.
    test_long_prefix_attention(target_seq_len=100_000)
