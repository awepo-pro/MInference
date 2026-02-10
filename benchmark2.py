# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from minference import MInference


def run_target_length(m: int, model, sampling_params, attn_type: str):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)

    s = 0
    T = 10
    for _ in range(T + 1):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = llm.generate([prompt], sampling_params)


        torch.cuda.synchronize()

        # warn up, first iteration _ := 0, indicate false
        if _:
            s += time.time() - start

        # avoid lazy 
        print(f'{outputs[0].outputs[0].text[:10]=}')
    print(attn_type, m, s / T)
    return s / T

def run_target_length2(m: int, model, sampling_params, attn_type: str):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)

    x = 500
    y = 3000
    warmup_prompt = prompt[:x]
    second_prompt = prompt[x:y]
    third_prompt = prompt[x:m]
    forth_prompt = prompt[x:2 * m]

    torch.cuda.synchronize()

    with torch.no_grad():
        outputs = llm.generate([warmup_prompt], sampling_params)

    torch.cuda.synchronize()
    print(f'{outputs[0].outputs[0].text[:10]=}')

    torch.cuda.synchronize()

    with torch.no_grad():
        outputs = llm.generate([second_prompt], sampling_params)

    torch.cuda.synchronize()
    print(f'{outputs[0].outputs[0].text[:10]=}')

    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        outputs = llm.generate([third_prompt, forth_prompt], sampling_params)

    torch.cuda.synchronize()
    used = time.time() - start
    print(f'{outputs[0].outputs[0].text[:10]=}')
    print(f'{outputs[1].outputs[0].text[:10]=}')
    print(attn_type, f'prefix: {len(second_prompt)}', f'compute: {len(third_prompt) - len(second_prompt)}', f'compute: {len(forth_prompt)}')
    print(f'time: {used}')

    return used


if __name__ == "__main__":
    torch.cuda.empty_cache()

    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        # default="./models--Qwen--Qwen2-0.5B",
        default="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    )
    args.add_argument(
        "--attn_type",
        type=str,
        # choices=["flash_attn", "minference"],
        default="minference"
    )
    args.add_argument("--context_window", type=int, default=100_000)
    args = args.parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=42
    )

    llm = LLM(
        model_name,
        enforce_eager=True,
        max_model_len=24576,        # 256 * 96
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        block_size=32,
        gpu_memory_utilization=0.95
    )

    # Patch MInference Module
    if args.attn_type == "minference":
        minference_patch = MInference("vllm_minference", model_name)
        llm = minference_patch(llm)

    run_target_length2(args.context_window, llm, sampling_params, args.attn_type)
