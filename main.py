# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from vllm import LLM, SamplingParams

from minference import MInference  # including MInference

def main() -> None:
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
    model_name = "./Llama-32-3B-tweets-10-adapt"
    llm = LLM(
        model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        max_model_len=128000,
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
