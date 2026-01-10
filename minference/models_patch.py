# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

from .minference_configuration import MInferenceConfig
from .patch import minference_patch_vllm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MInference:
    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_type: str = "dense",
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInference, self).__init__()
        self.config = MInferenceConfig(
            attn_type=attn_type,
            model_name=model_name,
            config_path=config_path,
            starting_layer=starting_layer,
            kv_cache_cpu=kv_cache_cpu,
            kv_type=kv_type,
            is_search=is_search,
            attn_kwargs=attn_kwargs,
            **kwargs,
        )

    def __call__(self, model):
        return self.patch_model(model)

    def patch_model(self, model):
        # KV type := dense (by deafult)

        # attention type := minference_patch_vllm (by default)
        if "vllm" not in self.config.attn_type:
            model.config.starting_layer = self.config.starting_layer
            model.config.config_path = self.config.config_path

        model = minference_patch_vllm(
            model, self.config.config_path, self.config.attn_kwargs
        )
        return model
