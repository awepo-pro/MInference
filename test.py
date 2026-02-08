from vllm import LLM, SamplingParams
import time

def run_prefix_caching_demo():
    model_name = "./models--Qwen--Qwen2-0.5B"

    llm = LLM(
        model=model_name, # Using a small model for demo purposes
        enable_prefix_caching=True,
        tensor_parallel_size=1,
        enforce_eager=True # disable CUDA graph
    )

    # Use greedy sampling (temperature=0) for stable benchmarks
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    # --- PHASE 0: WARM-UP (CRITICAL) ---
    print("\n--- Warming up (Ignore this time) ---")
    # This forces CUDA initialization, memory allocation, and graph capture.
    llm.generate(["Warm up the engine"], sampling_params)
    print("Engine is warm.")


    # --- Turn 1: The Initial Request ---
    with open('dataset/sonnets.txt', 'r') as input:
        data = input.read()[:10000]
        data1 = data[:8000]
        data2 = data[8000:10000]
    
    prompt_turn_1 = data1
    print(f'{len(prompt_turn_1)=}')
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1000)
    
    start_time = time.time()
    outputs_1 = llm.generate([prompt_turn_1], sampling_params)
    end_time = time.time()
    
    generated_text_1 = outputs_1[0].outputs[0].text
    print(f"Output 1: {generated_text_1[:50]}...")
    print(f"Turn 1 Time: {end_time - start_time:.4f}s (Cache Miss - Prefill required)")

    # --- Turn 2: Continuing the Session ---
    
    # CRITICAL STEP: 
    # To use the cache, we MUST start with the exact tokens from the previous turn.
    # We construct prompt_2 by appending the previous output + new user input.
    prompt_turn_2 = f"{prompt_turn_1}{generated_text_1}\n{data2}"
    print(f'{len(prompt_turn_2)=}')
    
    start_time = time.time()
    outputs_2 = llm.generate([prompt_turn_2], sampling_params)
    end_time = time.time()
    
    generated_text_2 = outputs_2[0].outputs[0].text
    print(f"Output 2: {generated_text_2[:50]}")
    print(f"Turn 2 Time: {end_time - start_time:.4f}s (Cache Hit - Prefill skipped for prefix)")

if __name__ == "__main__":
    run_prefix_caching_demo()
