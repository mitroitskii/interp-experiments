# %%
import os
import torch
import random
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# %%

# --- Constants ---
INIT_SEED = 42
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PROMPTS_FILE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_features/data/responses_deepseek-r1-distill-llama-8b.json"
OUTPUT_FILE = "new_generated_outputs.json"
SAMPLING_PARAMS = {
    "name": "DeepSeek recommended",
    "params": {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
    }
}
MAX_NEW_TOKENS = 32768 # longer context to get a full reasoning trace where possible
BATCH_SIZE = 16

# %%

# Load tokenizer for prompt formatting only
print(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=8,  # set >1 if using multiple GPUs
    dtype="bfloat16",
    gpu_memory_utilization=0.95
)

# %%

def generate_responses(batch_idx, messages, model, config, tokenizer):
    """
    Generates texts for a batch of messages, returns a list of responses.
    """
    print(f"--- Running batch {batch_idx} (size={len(messages)}) ---")

    # Format prompts using chat template (return strings for vLLM)
    formatted_prompts = [
        tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
        for msg in messages
    ]

    # Prepare sampling params for vLLM
    sampling_params = SamplingParams(
        seed=INIT_SEED,
        temperature=config["params"].get("temperature", 0.6),
        top_p=config["params"].get("top_p", 0.95),
        max_tokens=MAX_NEW_TOKENS,
    )

    # Run generation in a single batched call
    outputs = model.generate(prompts=formatted_prompts, sampling_params=sampling_params)
    return [o.outputs[0].text for o in outputs]


def load_prompts(file_path):
    """Loads prompts from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [item['original_message']['content'] for item in data]
    except (IOError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading or parsing prompts file {file_path}: {e}")
        return []

# %%

# Load prompts and generate responses
prompts = load_prompts(PROMPTS_FILE_PATH)
# Find next available filename if output file exists
output_file = OUTPUT_FILE
counter = 1
while os.path.exists(output_file):
    # Split filename and extension
    name, ext = os.path.splitext(OUTPUT_FILE)
    output_file = f"{name}_{counter}{ext}"
    counter += 1

if prompts:
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(prompts)} prompts in {total_batches} batches...")
    
    # Create new file at start of processing
    try:
        with open(output_file, 'w') as f:
            json.dump([], f)  # Initialize with empty list
        print(f"Created new output file: {output_file}")
    except Exception as e:
        print(f"Error creating output file {output_file}: {e}")
        exit(1)
    
    all_results = []
    
    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start:start + BATCH_SIZE]
        messages = [{"role": "user", "content": p} for p in batch_prompts]

        responses = generate_responses(
            batch_idx=start // BATCH_SIZE,
            messages=messages,
            model=model,
            config=SAMPLING_PARAMS,
            tokenizer=tokenizer,
        )

        for prompt_content, response in zip(batch_prompts, responses):
            result = {
                "model": MODEL_NAME,
                "sampling_params": SAMPLING_PARAMS["name"],
                "seed": INIT_SEED,
                "prompt": prompt_content,
                "response": response
            }
            all_results.append(result)

        # Save intermediate results after each batch
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved intermediate results after batch {start // BATCH_SIZE + 1}/{total_batches}")
        except Exception as e:
            print(f"Error saving intermediate results to {output_file}: {e}")

    print(f"\nAll results saved to {output_file}")
    print(f"Total prompts processed: {len(all_results)}")
else:
    print("No prompts found to process.")

# %%