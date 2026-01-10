# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from .configs.model2path import get_support_models

# flake8: noqa
from .minference_configuration import MInferenceConfig
from .models_patch import MInference
# from .ops.block_sparse_flash_attention import block_sparse_attention
# from .ops.pit_sparse_flash_attention_v2 import vertical_slash_sparse_attention
# from .patch import (
    # minference_patch,
    # minference_patch_kv_cache_cpu,
    # minference_patch_with_kvcompress,
# )
from .version import VERSION as __version__

__all__ = [
    "MInference",
    "MInferenceConfig",
    # "minference_patch",
    # "minference_patch_kv_cache_cpu",
    # "minference_patch_with_kvcompress",
    # "vertical_slash_sparse_attention",
    # "block_sparse_attention",
    "get_support_models",
]
