# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from vllm import LLM, SamplingParams

from minference import MInference  # including MInference

import os

def main() -> None:
    assert os.environ.get('VLLM_USE_V1') == '0', f'using v1, VLLM_USE_V1: {os.environ.get("VLLM_USE_V1")}'
    assert os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0', f'using multiprocessing, VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}'
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=10,
    )
    model_name = "./models--mlx-community--Llama-3.2-3B-Instruct"
    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        dtype='float16',
        max_model_len=12800,
    )

# Patch MInference Module
    minference_patch = MInference("vllm", model_name)
    llm = minference_patch(llm)

    outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
