# * TODO: record it in notion
import torch

x = torch.Tensor([
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
    [
        [1, 2, 3],
        [4, 5, 6]
        ]
    ], 
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    ]
])

print(x.shape)
print(x.stride())

a, b, c, d = x.shape
assert x.stride(3) == 1 
assert d == x.stride(2)
assert c * d == x.stride(1)
assert c * d * b == x.stride(0)
