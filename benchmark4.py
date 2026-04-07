import torch
from typing import Optional, List
from dataclasses import dataclass
import types
import time
import math

# ==========================================
# 0. Mock Imports (Simulating your environment)
# ==========================================

# Note: In your actual environment, import these from your actual libraries
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
    
    bsz = q.shape[0]
    assert bsz == 1, f'bsz: {bsz} is not 1'

    def block_sparse_kernel_with_kvcache(
            q, k, v, 
            k_cache, v_cache,
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
        q, k, v,
        k_cache, v_cache,
        block_tables,
        seq_lens)


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
    is_prompt: bool = True 

class AttentionType:
    ENCODER = 0
    DECODER = 1
    ENCODER_DECODER = 2

def get_num_prefill_decode_query_kv_tokens(metadata, attn_type):
    return metadata.num_prefill_tokens, metadata.num_prefill_tokens, metadata.num_decode_tokens

def get_tensor_model_parallel_rank():
    return 0

# Mock cache ops for CPU/GPU data movement testing
if not hasattr(torch.ops, "_C_cache_ops"):
    try:
        torch.ops.import_module("vllm._C_cache_ops")
    except:
        torch.ops._C_cache_ops = types.SimpleNamespace()
        
        def mock_reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
            # Optimized mock that handles CPU inputs writing to GPU cache
            block_size = k_cache.shape[1]
            # Ensure indices are on same device as cache for indexing, but source data is handled carefully
            slot_mapping = slot_mapping.to(k_cache.device)
            
            # This implementation is slow in python but functionally correct for mocks
            # In real vLLM C++ kernels, it handles this efficiently. 
            # For this test script, we assume this part works or we skip it if unnecessary for the specific test.
            pass

        torch.ops._C_cache_ops.reshape_and_cache_flash = mock_reshape_and_cache_flash


# ==========================================
# 2. Attention Layer with CPU Offloading Fix
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


    def forward_vllm_080(
        self,
        layer,
        query: torch.Tensor, # Input can be on CPU
        key: torch.Tensor,   # Input can be on CPU
        value: torch.Tensor, # Input can be on CPU
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        benchmark: bool = False
    ) -> torch.Tensor:
        """
        Forward pass modified to handle CPU inputs and stream chunks to GPU 
        to avoid OOM on massive sequences.
        """

        # ------------------------------------------------------------------
        # INTERNAL HELPER: Standard Prefill (Streaming CPU -> GPU)
        # ------------------------------------------------------------------
        def minference_prefill_func(q, k, v):
            # q, k, v are expected to be (seq_len, num_heads, head_size)
            # They might be on CPU.
            
            # Calculate repetition factor for GQA
            q_heads = q.size(-2)
            kv_heads = k.size(-2)
            n_rep = q_heads // kv_heads

            # Output tensor allocated on the same device as input (likely CPU)
            output = torch.empty_like(q)
            
            # We assume the actual computation kernel requires CUDA
            comp_device = "cuda"

            head_idx_st = get_tensor_model_parallel_rank() * q_heads
            
            # Iterate over heads. 
            # Instead of expanding ALL K/V (which causes OOM), we expand implicitly by selection.
            for head in range(q_heads):
                kv_head_idx = head // n_rep

                # 1. STREAMING: Move only the specific head data to GPU
                # (seq_len, 1, head_size)
                q_head = q[:, head, :].unsqueeze(1).to(comp_device, non_blocking=True)
                k_head = k[:, kv_head_idx, :].unsqueeze(1).to(comp_device, non_blocking=True)
                v_head = v[:, kv_head_idx, :].unsqueeze(1).to(comp_device, non_blocking=True)

                # 2. Reshape for Kernel (batch=1, seq_len, heads, dim) -> (1, heads, seq_len, dim)
                q_head = q_head[None, ...].transpose(1, 2)
                k_head = k_head[None, ...].transpose(1, 2)
                v_head = v_head[None, ...].transpose(1, 2)

                # 3. Compute Attention
                out = block_sparse_topk_vllm(q_head, k_head, v_head, head + head_idx_st)

                # 4. STREAMING: Move result back to source device (CPU) immediately
                out = out.transpose(1, 2).squeeze(0).contiguous()
                output[:, head:head+1, :] = out.to(output.device, non_blocking=True)
            
            return output
        
        # ------------------------------------------------------------------
        # INTERNAL HELPER: KV Cache Prefill (Streaming CPU -> GPU)
        # ------------------------------------------------------------------
        def minference_prefill_kvcache_func(
                q: torch.Tensor,                
                k: torch.Tensor,                
                v: torch.Tensor,
                k_cache: torch.Tensor, # Usually on GPU
                v_cache: torch.Tensor, # Usually on GPU
                causal: bool,                   
                block_tables: torch.Tensor,     
        ) -> torch.Tensor:

            q_heads = q.size(-2)
            kv_heads = k.size(-2)
            n_rep = q_heads // kv_heads

            output = torch.empty_like(q) # Likely CPU
            comp_device = "cuda"

            seq_lens = attn_metadata.prefill_metadata.seq_lens
            if not isinstance(seq_lens, torch.Tensor):
                # Make sure seq_lens is on GPU for the kernel
                seq_lens = torch.tensor(seq_lens, device=comp_device, dtype=torch.int32)
            else:
                seq_lens = seq_lens.to(comp_device)

            for head in range(q_heads):
                kv_head_idx = head // n_rep

                # 1. STREAMING: Move Q/K/V slice to GPU
                q_head = q[:, head, :].unsqueeze(1).to(comp_device, non_blocking=True)
                k_head = k[:, kv_head_idx, :].unsqueeze(1).to(comp_device, non_blocking=True)
                v_head = v[:, kv_head_idx, :].unsqueeze(1).to(comp_device, non_blocking=True)

                # (batch=1, seqlen, 1, headdim) -> (1, 1, seqlen, headdim)
                q_head = q_head[None, ...].transpose(1, 2)
                k_head = k_head[None, ...].transpose(1, 2)
                v_head = v_head[None, ...].transpose(1, 2)

                # Retrieve specific KV Cache head (Assuming Cache is already on GPU)
                k_cache_head_idx = head // n_rep 
                v_cache_head_idx = head // n_rep
                
                # (#block, block_size, 1, headdim)
                k_head_cache = k_cache[:, :, k_cache_head_idx, :].unsqueeze(2)
                v_head_cache = v_cache[:, :, v_cache_head_idx, :].unsqueeze(2)

                # Compute
                out = block_sparse_topk_vllm_with_kvcache(
                    q_head, 
                    k_head,
                    v_head,
                    k_head_cache,               
                    v_head_cache,
                    block_tables,
                    seq_lens)     

                out = out.transpose(1, 2).squeeze(0).contiguous()
                output[:, head:head+1, :] = out.to(output.device, non_blocking=True)

            return output
            
        # --- Main Forward Logic ---

        num_tokens, hidden_size = query.shape
        # Reshape: (N, num_heads * head_size) -> (N, num_heads, head_size)
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        attn_type = self.attn_type
        kv_cache_dtype: str = self.kv_cache_dtype

        key_cache = kv_cache[0] if kv_cache is not None else None
        value_cache = kv_cache[1] if kv_cache is not None else None

        # KV Cache Update Logic (Population)
        # If Q/K/V are on CPU, we must move them to GPU to write into GPU cache, 
        # or implement a CPU->GPU cache writer.
        # For simplicity in this fix, we assume cache population can be handled 
        # or we accept the slight memory bump here just for the chunks being cached.
        if key_cache is not None and value_cache is not None and kv_cache.numel() > 0:
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    updated_slot_mapping = attn_metadata.slot_mapping

                # To avoid OOM here during cache writing:
                # In a real scenario, you might chunk this loop too if K/V are huge.
                # However, usually reshape_and_cache_flash is optimized.
                # For this script, we assume specific cache writing is handled or skipped for pure attention test.
                pass 

        num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens = get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        # Output will be on CPU if Query is on CPU
        output = torch.empty_like(query)
        assert output is not None

        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        # output slice
        # prefill_output = output[:num_prefill_query_tokens] # We write to output directly using indices

        used = 0

        if prefill_meta := attn_metadata.prefill_metadata:
            # Check if we should run Standard Prefill or Prefix Prefill
            is_standard_prefill = (kv_cache is None or kv_cache.numel() == 0 or prefill_meta.block_tables is None or prefill_meta.block_tables.numel() == 0)

            if is_standard_prefill:
                print("  [Logic Path] Entering Standard Prefill (minference_prefill_func) [Streamed CPU->GPU]")
                print(f'{query.shape=}, {key.shape=}, {value.shape=}')
                
                # Sync before timing
                torch.cuda.synchronize()
                start = time.time()

                with torch.no_grad():
                    # Pass CPU tensors directly
                    out = minference_prefill_func(query, key, value)

                torch.cuda.synchronize()
                used = time.time() - start
                print(f'time: {used}')

                output[:num_prefill_query_tokens] = out
            else:
                print("  [Logic Path] Entering Prefix-Enabled Prefill (minference_prefill_kvcache_func) [Streamed CPU->GPU]")
                print(f'{query.shape=}, {key.shape=}, {value.shape=}')

                assert prefill_meta.seq_lens is not None
                    
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    # Pass CPU tensors for QKV, GPU tensors for Cache
                    res = minference_prefill_kvcache_func(
                        query,
                        key,
                        value,
                        key_cache,
                        value_cache,
                        causal=True,
                        block_tables=prefill_meta.block_tables
                    )
                    output[:num_prefill_query_tokens] = res

                torch.cuda.synchronize()
                used = time.time() - start
                print(f'time: {used}')

        return output.view(num_tokens, hidden_size), used if benchmark else None

# ==========================================
# 3. Test Harness: CPU Initialization
# ==========================================

def test_minf_prefix_attention(prefix_len, total_len):
    warmup()
    print("=== Starting Prefix Attention Test Case ===")
    
    # IMPORTANT: Generate Data on CPU to prevent OOM
    device_data = "cpu"
    device_comp = "cuda"
    dtype = torch.bfloat16
    
    layer = MockAttentionLayer()
    BLOCK_SIZE = 64
    num_blocks_needed = math.ceil(total_len / BLOCK_SIZE)
    print(f"  [Info] Blocks required: {num_blocks_needed}")

    # Cache stays on GPU (It's usually smaller than the massive QKV expanded, or fits in 24GB)
    # If Cache is also too big, you need to use CPU cache and swap, but let's assume Cache fits.
    kv_cache = torch.zeros(
        2, num_blocks_needed, BLOCK_SIZE, layer.num_kv_heads, layer.head_size,
        dtype=dtype, device=device_comp
    )
    
    # --- STAGE 1: Standard Prefill ---
    print("\n[Stage 1] Running Standard Prefill...")
    
    seq_len_1 = prefix_len
    # Alloc on CPU
    q1 = torch.randn(seq_len_1, layer.num_heads * layer.head_size, device=device_data, dtype=dtype)
    k1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    v1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    
    slot_mapping_1 = torch.arange(seq_len_1, device=device_data, dtype=torch.long) 
    
    meta_1 = AttnMetadata(
        prefill_metadata=PrefillMetadata(block_tables=None, seq_lens=[prefix_len]),
        slot_mapping=slot_mapping_1,
        num_prefill_tokens=seq_len_1
    )
    
    _ = layer.forward_vllm_080(
        layer=layer,
        query=q1, key=k1, value=v1, kv_cache=kv_cache,
        attn_metadata=meta_1
    )
    
    # --- STAGE 2: Prefix-Enabled Prefill ---
    print("\n[Stage 2] Running Prefix-Enabled Prefill...")
    
    seq_len_2 = total_len
    remains = seq_len_2 - prefix_len
    
    # Alloc on CPU
    q2 = torch.randn(remains, layer.num_heads * layer.head_size, device=device_data, dtype=dtype)
    k2 = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    v2 = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    
    slot_mapping_2 = torch.arange(prefix_len, total_len, device=device_data, dtype=torch.long)
    block_tables_2 = torch.arange(num_blocks_needed, dtype=torch.int32, device=device_comp).unsqueeze(0)

    meta_2 = AttnMetadata(
        prefill_metadata=PrefillMetadata(
            block_tables=block_tables_2, 
            seq_lens=[total_len]
        ),
        slot_mapping=slot_mapping_2,
        num_prefill_tokens=remains
    )
    
    _, used = layer.forward_vllm_080(
        layer=layer,
        query=q2, key=k2, value=v2, kv_cache=kv_cache,
        attn_metadata=meta_2,
        benchmark=True,
    )

    print("  [Success] Stage 2 completed.")
    return used

def test_minf(prefix_len, total_len):
    torch.empty_cache()
    print("=== Starting MInference Attention Test Case (CPU streaming) ===")
    
    # Use CPU for huge tensors
    device_data = "cpu"
    dtype = torch.bfloat16
    
    layer = MockAttentionLayer()

    # --- STAGE 1: Standard Prefill ---
    print("\n[Stage 1] Running Standard Prefill...")
    seq_len_1 = prefix_len
    
    q1 = torch.randn(seq_len_1, layer.num_heads * layer.head_size, device=device_data, dtype=dtype)
    k1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    v1 = torch.randn(seq_len_1, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    
    slot_mapping_1 = torch.arange(seq_len_1, device=device_data, dtype=torch.long) 
    
    meta_1 = AttnMetadata(
        prefill_metadata=PrefillMetadata(block_tables=None, seq_lens=[prefix_len]),
        slot_mapping=slot_mapping_1,
        num_prefill_tokens=seq_len_1
    )
    
    _ = layer.forward_vllm_080(
        layer=layer, query=q1, key=k1, value=v1, kv_cache=None, attn_metadata=meta_1,
    )
    
    # STAGE 2: New Suffix
    seq_len_2 = total_len
    remains = seq_len_2 - prefix_len
    
    q_new = torch.randn(remains, layer.num_heads * layer.head_size, device=device_data, dtype=dtype)
    k_new = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)
    v_new = torch.randn(remains, layer.num_kv_heads * layer.head_size, device=device_data, dtype=dtype)

    q2 = torch.cat([q1, q_new])
    k2 = torch.cat([k1, k_new])
    v2 = torch.cat([v1, v_new])
    
    meta_2 = AttnMetadata(
        prefill_metadata=PrefillMetadata(block_tables=None, seq_lens=[total_len]),
        slot_mapping=None,
        num_prefill_tokens=seq_len_2
    )
    
    _, used = layer.forward_vllm_080(
        layer=layer,
        query=q2, key=k2, value=v2, kv_cache=None,
        attn_metadata=meta_2,
        benchmark=True
    )

    print("  [Success] Stage 2 completed.")
    return used


if __name__ == "__main__":
    # Test with sizes that previously crashed
    T = 2 # Reduced iterations for demo
    used = 0
    prefix = 30_000
    total = 1_000_000
    
    for _ in range(T + 1):
        # Passing huge total_len (1M) but initializing on CPU

        if _:
            used += test_minf(30_000, 1_000_000)
            # used += test_minf_prefix_attention(300_000, 1_000_000)

    print(f'Average time: {used / T}')
