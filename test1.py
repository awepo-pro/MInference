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
):
    head_id = 0

    head_offset = head_id * s_num_head

    bt_id = 3

    bt_offset = bt_id * s_num_block

    block_id = 0

    block_offset = block_id * s_block_size + tl.arange(0, BLOCK_SIZE_N)

    dim_offset = tl.arange(0, BLOCK_DIM) * s_head_dim

    dst = cache + bt_offset + head_offset + block_offset + dim_offset 

    k = tl.load(dst)

    tl.store(o, k)


def arange_allocate(size):
    x = torch.zeros(size, dtype=torch.float16)
    x = torch.arange(1, x.numel() + 1, dtype=torch.float16).reshape(x.shape)

    return x 



if __name__ == '__main__':

    cache = arange_allocate((10, 16, 2, 64))

    cache_head = cache[:, :, 0:1, :]

    o = torch.empty_like(cache_head)

    print(cache_head.shape)
    print(cache_head)
    print(o.shape)
    s_num_block, s_block_size, s_num_head, s_head_dim = cache_head.stride()
    
    grid = (1, 1, 1)
    
    assert False
    test[grid](
        cache_head, 
        o,
        s_num_block,
        s_block_size,
        s_num_head,
        s_head_dim)
