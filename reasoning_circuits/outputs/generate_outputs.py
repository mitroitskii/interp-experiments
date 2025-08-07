# %%
import os
import torch
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# %%

# --- Constants ---
INIT_SEED = 42
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PROMPTS_FILE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_features/data/responses_deepseek-r1-distill-llama-8b.json"
OUTPUT_FILE = "new_generated_outputs.json"
GENERATION_CONFIG = {
    "name": "Sampling (DeepSeek recommended)",
    "params": {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
    }
}
MAX_NEW_TOKENS = 1000

# %%

# Initial seed setting
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)

# Load model and tokenizer
print(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Fix pad token issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# %%


def generate_response(counter, message, model, tokenizer, config):
    """
    Generates text for a given seed and message, returns the response.
    """
    print(f"--- Running generation {counter} ---")

    set_seed(INIT_SEED)
    random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)

    # Prepare the input message with attention mask
    inputs = tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, return_tensors="pt")
    input_ids = inputs.to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)

    # Run generation
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        **config["params"]
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return response


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
all_results = []

if prompts:
    for counter, prompt_content in enumerate(prompts):
        message = {"role": "user", "content": prompt_content}
        response = generate_response(counter, message, model, tokenizer, GENERATION_CONFIG)

        result = {
            "model": MODEL_NAME,
            "generation_config": GENERATION_CONFIG["name"],
            "seed": INIT_SEED,
            "prompt": prompt_content,
            "response": response
        }
        all_results.append(result)

    # Save all results to JSON file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {OUTPUT_FILE}")
        print(f"Total prompts processed: {len(all_results)}")
    except Exception as e:
        print(f"Error saving results to {OUTPUT_FILE}: {e}")
else:
    print("No prompts found to process.")

# %%
