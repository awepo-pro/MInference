import torch
import triton
import triton.language as tl




@triton.jit
def test(
    cache,
    o,
    s_num_block,
    s_block_size,
    s_num_head,
    s_head_dim,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    dtype: tl.constexpr,
):
    bt_id = 3
    bt_offset = bt_id * s_num_block

    head_id = 0
    head_offset = head_id * s_num_head

    block_id = 0
    block_offset = block_id * s_block_size + tl.arange(0, BLOCK_SIZE_N)

    dim_offset = 0 * s_head_dim + tl.arange(0, BLOCK_DIM)

    dst = cache + bt_offset + head_offset + block_offset + dim_offset 
    k = tl.load(dst)

    tl.store(o, k.to(dtype))


def arange_allocate(size):
    x = torch.zeros(size, dtype=torch.int32)
    x = torch.arange(1, x.numel() + 1, dtype=torch.int32).reshape(x.shape)

    return x 



if __name__ == '__main__':

    # * (#b, b_size, #head, headdim)
    cache = arange_allocate((10, 16, 2, 64))
    # cache = arange_allocate((1, 2, 3, 4))

    cache_head = cache[:, :, 0:1, :]
    print(cache)

    o = torch.empty_like(cache_head)

    print(cache_head.shape)
    print(cache_head)
    print(o.shape)
    s_num_block, s_block_size, s_num_head, s_head_dim = cache_head.stride()
    
    grid = (1, 1, 1)
    
    test[grid](
        cache_head, 
        o,
        s_num_block,
        s_block_size,
        s_num_head,
        s_head_dim,
        8,
        64,
        cache.dtype)
