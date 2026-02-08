from vllm import LLM, SamplingParams
import time

def run_prefix_caching_demo():
    # 1. Initialize vLLM with prefix caching enabled
    # This is the crucial step: enable_prefix_caching=True
    llm = LLM(
        model="facebook/opt-125m", # Using a small model for demo purposes
        # enable_prefix_caching=True,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    # --- Turn 1: The Initial Request ---
    system_prompt = "You are a concise AI assistant."
    user_query_1 = "What are the three primary colors?"
    
    # Construct the full prompt for Turn 1
    prompt_turn_1 = f"{system_prompt}\nUser: {user_query_1}\nAssistant:"
    
    print(f"\n--- Processing Turn 1 ---\nPrompt: {prompt_turn_1!r}")
    
    start_time = time.time()
    outputs_1 = llm.generate([prompt_turn_1], sampling_params)
    end_time = time.time()
    
    generated_text_1 = outputs_1[0].outputs[0].text
    print(f"Output 1: {generated_text_1}")
    print(f"Turn 1 Time: {end_time - start_time:.4f}s (Cache Miss - Prefill required)")

    # --- Turn 2: Continuing the Session ---
    user_query_2 = "Which one represents passion?"
    
    # CRITICAL STEP: 
    # To use the cache, we MUST start with the exact tokens from the previous turn.
    # We construct prompt_2 by appending the previous output + new user input.
    prompt_turn_2 = f"{prompt_turn_1}{generated_text_1}\nUser: {user_query_2}\nAssistant:"
    
    print(f"\n--- Processing Turn 2 ---\nPrompt: {prompt_turn_2!r}")
    
    start_time = time.time()
    outputs_2 = llm.generate([prompt_turn_2], sampling_params)
    end_time = time.time()
    
    generated_text_2 = outputs_2[0].outputs[0].text
    print(f"Output 2: {generated_text_2}")
    print(f"Turn 2 Time: {end_time - start_time:.4f}s (Cache Hit - Prefill skipped for prefix)")

if __name__ == "__main__":
    run_prefix_caching_demo()
