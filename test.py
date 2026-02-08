from vllm import LLM, SamplingParams
from minference import MInference
import time

def brenchmark(llm, prompts):
    minference_patch = MInference("vllm_minference", model_name)
    llm = minference_patch(llm)

    # Use greedy sampling (temperature=0) for stable benchmarks
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # --- PHASE 0: WARM-UP (CRITICAL) ---
    print("\n--- Warming up (Ignore this time) ---")
    # This forces CUDA initialization, memory allocation, and graph capture.
    llm.generate(["Warm up the engine"], sampling_params)
    print("Engine is warm.")

    acc_prompt = ''
    time_used = []

    for i, prompt in enumerate(prompts):
        print(f'request {i}, len: {len(prompt)}')
        
        sampling_params = SamplingParams(temperature=0, max_tokens=1)
        
        acc_prompt = acc_prompt + prompt
        start_time = time.time()
        output = llm.generate([acc_prompt], sampling_params)
        end_time = time.time()
        
        # output[0] only one question, so must always index 0
        generated_text = output[0].outputs[0].text
        print(f"Output: {generated_text[:50]}...")
        # print(f"Turn Time: {end_time - start_time:.4f}s (Cache Miss - Prefill required)")

        acc_prompt = acc_prompt + '\n' +  generated_text
        time_used.append(end_time - start_time)

    return time_used

if __name__ == "__main__":

    with open('dataset/sonnets.txt', 'r') as input:
        data = input.read()[:10000]
        data1 = data[:8000]
        data2 = data[8000:10000]

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=42
    )

    model_name = "./models--Qwen--Qwen2-0.5B"
    
    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        dtype='float16',
        max_model_len=12800,
        block_size=256,
        enable_prefix_caching=True,
    )

    with_prefix_t = brenchmark(llm, [data1, data2])

    llm = LLM(
        model=model_name,
        max_num_seqs=1,
        enforce_eager=True,     # disable to get 2-3x faster speed for CUDA graph
        dtype='float16',
        max_model_len=12800,
        block_size=256,
        # enable_prefix_caching=True,
    )

    without_prefix_t = brenchmark(llm, [data1, data2])

    print(f'standard minfernece: {without_prefix_t}')
    print(f'with prefix enable minference: {with_prefix_t}')
    
