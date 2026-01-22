import torch
import triton
import triton.language as tl

@triton.jit
def test(
    cache_ptr,      # (#block, block_size, #head, headdim)
    o_ptr,
    s_num_block,
    s_block_size,
    s_num_head,
    s_head_dim,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    dtype: tl.constexpr,
):
    # assume we want to locate half of the block table in block tables 
    bt_id = 1.5
    block_id = 1
    head_id = 0

    # use index to locate the place
    bt_offset = bt_id * s_num_block
    head_offset = head_id * s_num_head

    # use offset to locate every elements, tl.arange used for range
    block_offset = (s_block_size * tl.arange(0, BLOCK_SIZE_N))[:, None]
    dim_offset = (s_head_dim * tl.arange(0, BLOCK_DIM))[None, :]

    dst_ptr = cache_ptr + bt_offset.to(tl.int32) + block_offset + head_offset + dim_offset 
    k = tl.load(dst_ptr)

    # 1 head is hidden in cache_head, offset divide 2 times to ensure same place in o_ptr
    tl.store(o_ptr + block_offset // 2 + dim_offset, k.to(dtype))


def arange_allocate(size, dtype=torch.int32):
    # Use float32 or half for Triton kernels usually, but sticking to your int32 setup
    x = torch.arange(1, torch.Size(size).numel() + 1, dtype=dtype).reshape(size).cuda()
    return x 

if __name__ == '__main__':
    device = torch.device('cuda')

    # (#b, b_size, #head, headdim) -> (10, 16, 2, 64)
    cache = arange_allocate((3, 4, 2, 8))
    
    # Selecting a specific head: (10, 16, 1, 64)
    cache_head = cache[:, :, 0:1, :]
    
    # Prepare output buffer
    o = torch.zeros_like(cache_head)

    # Get strides
    s_num_block, s_block_size, s_num_head, s_head_dim = cache_head.stride()
    assert o.shape == cache_head.shape, f'{o.shape=} != {cache_head.shape=}'
    
    print(o.stride())
    print(o.shape)
    print(cache_head.stride())
    print(cache.stride())
    grid = (1, 1, 1)

    test[grid](
        cache_head, 
        o,
        s_num_block,
        s_block_size,
        s_num_head,
        s_head_dim,
        BLOCK_SIZE_N=cache_head.shape[1] // 2,
        BLOCK_DIM=cache_head.shape[-1],
        dtype=tl.int32      # doesn't use cache.dtype, troch.int32 != tl.int32
    )

    # Verification
    print("Verification (First 5 elements of loaded block):")
    print(cache_head)
    print(o)
