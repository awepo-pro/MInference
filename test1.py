import torch
import triton
import triton.language as tl

@triton.jit
def test(
    cache_ptr,
    o_ptr,
    s_num_block,
    s_block_size,
    s_num_head,
    s_head_dim,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    dtype: tl.constexpr,
):
    # Program IDs / Constants
    bt_id = 3
    head_id = 0
    block_id = 0

    # Calculate base offsets
    bt_offset = bt_id * s_num_block
    head_offset = head_id * s_num_head
    
    # Create 2D range offsets for broadcasting
    # block_offset shape: (BLOCK_SIZE_N, 1) -> (8, 1)
    block_offset = (block_id * s_block_size + tl.arange(0, BLOCK_SIZE_N))[:, None]
    
    # dim_offset shape: (1, BLOCK_DIM) -> (1, 64)
    dim_offset = (0 * s_head_dim + tl.arange(0, BLOCK_DIM))[None, :]

    # dst shape: (8, 64)
    # Triton handles the pointer arithmetic by broadcasting the addition
    dst_ptr = cache_ptr + bt_offset + head_offset + block_offset + dim_offset 
    
    # Load the 2D block
    k = tl.load(dst_ptr)

    # Store the 2D block into output
    # Note: o_ptr needs to be indexed similarly if you want to write to a specific location
    tl.store(o_ptr + block_offset + dim_offset, k.to(dtype))


def arange_allocate(size):
    # Use float32 or half for Triton kernels usually, but sticking to your int32 setup
    x = torch.arange(1, torch.Size(size).numel() + 1, dtype=torch.int32).reshape(size).cuda()
    return x 

if __name__ == '__main__':
    # Ensure we are on CUDA
    device = torch.device('cuda')

    # (#b, b_size, #head, headdim) -> (10, 16, 2, 64)
    cache = arange_allocate((10, 16, 2, 64))
    
    # Selecting a specific head: (10, 16, 1, 64)
    cache_head = cache[:, :, 0:1, :]
    
    # Prepare output buffer
    o = torch.zeros_like(cache_head)

    # Get strides
    s_num_block, s_block_size, s_num_head, s_head_dim = cache_head.stride()
    
    grid = (1, 1, 1)
    
    test[grid](
        cache_head, 
        o,
        s_num_block,
        s_block_size,
        s_num_head,
        s_head_dim,
        BLOCK_SIZE_N=8,
        BLOCK_DIM=64,
        dtype=tl.int32 # Adjusted to match cache.dtype
    )

    # Verify the first 8 elements of the 4th batch (bt_id=3) were copied
    print("Verification (First 5 elements of loaded block):")
    print(o[3, 0, 0, :5])
    print(cache[3, 0, 0, :5])
