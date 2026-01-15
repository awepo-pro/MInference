# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from vllm import LLM, SamplingParams

from minference import MInference  # including MInference

import os

def main() -> None:
    assert os.environ.get('VLLM_USE_V1') == '0', f'using v1, VLLM_USE_V1: {os.environ.get("VLLM_USE_V1")}'
    assert os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0', f'using multiprocessing, VLLM_ENABLE_V1_MULTIPROCESSING: {os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}'
    prompts = [
        # "Hello my name is",    # * 4 words
        # "The president of the United States is",    # * 7 words
        # "The capital of France is cold",    # * 6 words
        # "The future of AI is",      # * 5 words
        "Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592 he began a successful career in London as an actor, writer, and part-owner (\"sharer\") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613) he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories as to whether the works attributed to him were written by others.",
        "Shakespeare was born and raised in Stratford-upon-Avon, Warwickshire. At the age of 18, he married Anne Hathaway, with whom he had three children: Susanna, and twins Hamnet and Judith. Sometime between 1585 and 1592 he began a successful career in London as an actor, writer, and part-owner (\"sharer\") of a playing company called the Lord Chamberlain's Men, later known as the King's Men after the ascension of King James VI of Scotland to the English throne. At age 49 (around 1613) he appears to have retired to Stratford, where he died three years later. Few records of Shakespeare's private life survive; this has stimulated considerable speculation about such matters as his physical appearance, his sexuality, his religious beliefs and even certain fringe theories as to whether the works attributed to him were written by others."
    ]

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=10,
    )

    model_name = "./models--Qwen--Qwen2-0.5B"
    
    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        dtype='float16',
        max_model_len=12800,
        block_size=256,
    )

    # Patch MInference Module
    minference_patch = MInference("vllm_minference", model_name)
    llm = minference_patch(llm)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    main()
