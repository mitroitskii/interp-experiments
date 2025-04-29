# %%
import os
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# %%

# Initial seed setting (can be removed if not needed outside the loop)
INIT_SEED = 42
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)

# Load model and tokenizer
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
print(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# %%

# Define generation configurations (outside the function as it's constant)
generation_configs = [
    # { # Greedy
    #     "name": "Greedy",
    #     "params": {
    #         "do_sample": False,
    #         "temperature": None,
    #         "top_p": None,
    #     }
    # },
    # { # Produces equivalent results to deepseek recommended
    #     "name": "Sampling (default, top_k=0)",
    #     "params": {
    #         "do_sample": True,
    #         "top_k": 0,
    #     }
    # },
    # { # Produces equivalent results to deepseek recommended
    #     "name": "Sampling (temp=0.6)",
    #     "params": {
    #         "do_sample": True,
    #         "temperature": 0.6,
    #     }
    # },
    {
        "name": "Sampling (temp=1.0)",
        "params": {
            "do_sample": True,
            "temperature": 1.0,
        }
    },
    {
        "name": "Sampling (DeepSeek recommended)",
        "params": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
        }
    },
]

# Define output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


def generate_and_save(seed, message, model, tokenizer, generation_configs, output_dir):
    """
    Generates text based on different configurations for a given seed and message,
    and saves the results to a file. Does not print generated text to console.
    """
    print(f"--- Running generation for seed {seed} ---")
    # Prepare the input message
    input_ids = tokenizer.apply_chat_template(
        [message], add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Store results for this seed
    results = {}

    # Run generations
    for config in generation_configs:
        # Reset seed for each generation config for reproducibility within this seed run
        set_seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        outputs = model.generate(
            input_ids,
            max_new_tokens=1000,
            pad_token_id=tokenizer.eos_token_id,
            **config["params"]  # Unpack parameters
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Store the decoded text (tokens can be large, only storing text)
        results[config['name']] = response  # Only storing text results now

    # Save results to a file
    output_file = os.path.join(output_dir, f"seed_{seed}_results.txt")
    try:
        with open(output_file, "w") as f:
            f.write(f"Results for seed {seed}\n")
            f.write(f"Model: {MODEL_NAME}\n\n")
            f.write(f"Input message: {message['content']}\n\n")

            for config_name, result_text in results.items():
                f.write(f"--- {config_name} ---\n")
                f.write(result_text)
                f.write("\n\n")

            f.write("=" * 80 + "\n")
        print(f"Results for seed {seed} saved to {output_file}")
    except Exception as e:
        print(f"Error saving results for seed {seed} to {output_file}: {e}")
    
    torch.cuda.empty_cache()


# %%

# Define the constant input message
# prompt_1 = {"role": "user",
#                  "content": "Find three prime numbers that add up to 100."}

# prompt_2 =   {
#       "role": "user",
#       "content": "In a group of 5 people, what's the probability at least 2 were born in the same month?"}

# prompt_3 = {
#     "role": "user",
#     "content": "What's the probability of getting exactly one 6 when rolling five dice?"}

prompt_4 = {
      "role": "user",
      "content": "What's the sum of all proper divisors of 36?"
    }


# Loop over seeds and run generation
for seed_val in range(1, 42):  # Seeds 0 to 41
    generate_and_save(seed_val, prompt_4, model,
                      tokenizer, generation_configs, output_dir)

print("\n--- All seeds processed ---")