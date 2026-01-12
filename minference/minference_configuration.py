# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from .configs.model2path import LEANKPATNS, MODEL2PATH


class MInferenceConfig:
    MINFERENCE_ATTENTION_TYPES = [
        "minference",
        "vllm_minference",
        "tri_mix_minference",
    ]
    OTHER_ATTENTION_TYPES = [
        # original implement
        "hf",
        "vllm",
        # our custom implement
        "dense",
        "static",  # minference w/ static
        "dilated1",
        "dilated2",
        "a_shape",
        "tri_shape",
        "vllm_a_shape",
        "vllm_tri_shape",
        "inf_llm",
        "flexprefill",
        "vllm_flexprefill",
        "xattention",
        "tri_mix",
    ]
    KV_TYPES = [
        "dense",
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "quest",
        "retr_attn",
        "kivi",
        "leank",
    ]

    def __init__(
        self,
        attn_type: str = "minference",
        model_name: str = None,
        config_path: str = None,
        starting_layer: int = -1,
        kv_cache_cpu: bool = False,
        kv_cache_cpu_device: str = "cpu",
        kv_type: str = "dense",
        is_search: bool = False,
        attn_kwargs: dict = {},
        **kwargs,
    ):
        super(MInferenceConfig, self).__init__()
        # * alias names
        attn_type, kv_type = self.update_config_type(attn_type, kv_type)

        # * attn_type must in supported attn_type list
        assert (
            attn_type in self.MINFERENCE_ATTENTION_TYPES + self.OTHER_ATTENTION_TYPES
        ), f"The attn_type {attn_type} you specified is not supported."

        # * kv_type must in supported KV_type list
        assert (
            kv_type in self.KV_TYPES
        ), f"The kv_type {kv_type} you specified is not supported."

        print(f"<---- MInference Config Detail ----> attn_type {attn_type}, kv_type {kv_type}")

        self.attn_type = attn_type
        # * if model is supported (official), link to default config file
        self.config_path = self.update_config_path(config_path, model_name)
        self.model_name = model_name
        self.is_search = is_search
        self.starting_layer = starting_layer
        self.kv_cache_cpu = kv_cache_cpu
        self.kv_cache_cpu_device = kv_cache_cpu_device
        self.kv_type = kv_type
        self.attn_kwargs = {
            "is_search": is_search,
            "starting_layer": starting_layer,
            "config_path": config_path,
            **attn_kwargs,
        }

        assert kv_type != "leank", f'kv_type ({kv_type}) cannot be leank'
        if kv_type == "leank":
            model_name = model_name.split("/")[-1]
            self.leank_path = LEANKPATNS[model_name]

    def update_config_path(self, config_path: str = None, model_name: str = None):
        # * auto redirect if supported model
        if self.attn_type in self.OTHER_ATTENTION_TYPES:
            return ""
        
        # * use own config  
        if config_path is not None:
            return config_path

        # TODO: config is updated, update corresponding setting in codes
        assert (
            model_name in MODEL2PATH
        ), f"The model {model_name} you specified is not supported. You are welcome to add it and open a PR :)"

        return MODEL2PATH[model_name]

    def get(self, attr, default=None):
        return getattr(self, attr, default)

    def update_config_type(self, attn_type: str, kv_type: str):
        if kv_type == "":
            kv_type = "dense"
        if attn_type == "minference_with_dense":
            attn_type = "dense"
        return attn_type, kv_type

    @classmethod
    def get_available_attn_types(cls):
        return cls.MINFERENCE_ATTENTION_TYPES + cls.OTHER_ATTENTION_TYPES

    @classmethod
    def get_available_kv_types(cls):
        return cls.KV_TYPES
