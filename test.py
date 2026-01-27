import torch
from po_debug import debug_print

# x = torch.zeros((1, 10), dtype=torch.float16)
# x = torch.arange(1, x.numel() + 1, dtype=torch.float16).reshape(x.shape)

# debug_print(x)
# debug_print(x.shape)
# stride_a, stride_b = x.stride()

# debug_print(stride_a)
# debug_print(stride_b)

# y = x.flatten()
# debug_print(y)
# debug_print(y.shape)


# base_ptr = 0 * stride_a + 2 * stride_b

# debug_print(y[base_ptr])

# z = x.T

# debug_print(z, comment="before flatten")

# z = z.flatten()
# debug_print(z, comment="after flatten")
# debug_print(z.shape)
# debug_print(z[base_ptr])


# x = torch.zeros((1, 10), dtype=torch.float16)
# x = torch.arange(1, x.numel() + 1, dtype=torch.float16).reshape(x.shape)

# y = torch.zeros((5, 1), dtype=torch.float16)
# y = torch.arange(1, y.numel() + 1, dtype=torch.float16).reshape(y.shape)

# print(x)
# print(y)

# z = x + y
# print(z.shape)

# print(z)

# print(x)
# print(x.shape)
# print(x.stride())

# xa, xb, xc = x.stride()
# idx = 0 * xa + 1 * xb + 5 * xc
# print(f'{x.flatten()[idx]=}')

# idx2 = 0 * xa +  (1 * xb) // 3 + 5 * xc
# print(f'{idx2=}')
# y = x[:, 1:2, :]

# print(y)
# print(y.shape)
# print(y.stride())
# print(y.flatten()[idx2])

BLOCK_M = 31
BLOCK_DMODEL = 64
a = torch.arange(0, BLOCK_M)[:, None]   # * (1, BLOCK_M)
b = (torch.arange(0, BLOCK_DMODEL) * BLOCK_M)[None, :]      # * (BLOCK_DMODEL, 1)
print(a)
print(b)
offset_acc = a + b
print(offset_acc)
