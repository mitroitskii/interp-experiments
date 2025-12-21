# %%
# --- Imports ---
from typing import Callable, TypeVar
from functools import partial
import os
import json
import random
import gc
import einops
import torch
import torch.nn.functional as F
import nnsight as ns
from nnsight import LanguageModel
from transformers import AutoTokenizer, set_seed
from dictionary_learning import BatchTopKCrossCoder, CrossCoder

TT = TypeVar("TT")

# %%
# --- Constants ---
BATCH_SIZE = 1

GENERATION_SEED = 3
INIT_SEED = 42

DATA_SOURCE = "outputs"
# DATA_SOURCE = "data"

DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/outputs/new_generated_outputs.json" # prompt + response containing the full response with closing </think> token
SAVE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/results"

# --- Model Identifiers ---
REASONING_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ORIGINAL_BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# --- Configuration for current run ---
# Model to perform attribution on
# ATTRIBUTION_MODEL_CONFIG_ID = REASONING_MODEL_ID
ATTRIBUTION_MODEL_CONFIG_ID = ORIGINAL_BASE_MODEL_ID  # For next run with base model

# Model to use as the first input ('reference') for the CrossCoder
# This is typically the model the CrossCoder's first input head was trained on.
CROSSCODER_REFERENCE_MODEL_CONFIG_ID = ORIGINAL_BASE_MODEL_ID

# Short name for the attribution model, used in save paths
if ATTRIBUTION_MODEL_CONFIG_ID == REASONING_MODEL_ID:
    ATTRIBUTION_MODEL_SHORT_NAME = "Reasoning"
elif ATTRIBUTION_MODEL_CONFIG_ID == ORIGINAL_BASE_MODEL_ID:
    ATTRIBUTION_MODEL_SHORT_NAME = "Base"
else:
    # Fallback if a different model ID is used directly
    ATTRIBUTION_MODEL_SHORT_NAME = ATTRIBUTION_MODEL_CONFIG_ID.split(
        '/')[-1].replace('-', '_')

# Model to load tokenizer from (usually compatible one, e.g., reasoning model's)
TOKENIZER_MODEL_ID = REASONING_MODEL_ID

CROSSCODER_TYPE = "L1"
# CROSSCODER_TYPE = "BatchTopK"
CROSSCODER_LAYER = 7
CROSSCODER_DEVICE = "cuda:7"
CROSSCODER_PATH = f"/disk/u/troitskiid/data/checkpoints/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/cc_weights.pt"

BASELINE = 0

# %%
# --- Load Model and Tokenizer ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

print(f"Loading model for attribution: {ATTRIBUTION_MODEL_CONFIG_ID}...")
attribution_model_loaded = LanguageModel(
    ATTRIBUTION_MODEL_CONFIG_ID, device_map="auto", torch_dtype=torch.bfloat16)

if ATTRIBUTION_MODEL_CONFIG_ID == CROSSCODER_REFERENCE_MODEL_CONFIG_ID:
    crosscoder_reference_model_loaded = attribution_model_loaded
    print(f"Using attribution model as crosscoder reference model.")
else:
    print(
        f"Loading crosscoder reference model: {CROSSCODER_REFERENCE_MODEL_CONFIG_ID}...")
    crosscoder_reference_model_loaded = LanguageModel(
        CROSSCODER_REFERENCE_MODEL_CONFIG_ID, device_map="auto", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL_ID,  # Using the specified model's tokenizer
    use_fast=True,
    add_bos_token=True,
)
print("Models and Tokenizer loaded.")

# %%
# --- Load CROSSCODER ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

print(
    f"Loading {'/'.join(CROSSCODER_PATH.split('/')[-3:-2])} for layer {CROSSCODER_LAYER}...")

if CROSSCODER_TYPE == "BatchTopK":
    cc = BatchTopKCrossCoder
else:
    cc = CrossCoder

crosscoder = cc.from_pretrained(
    path=CROSSCODER_PATH,
    dtype=torch.bfloat16,
    device=CROSSCODER_DEVICE,
    from_hub=False,
)
print("CROSSCODER loaded.")


# %%
# --- Load Data ---
print(f"Loading data from {DATA_PATH}...")
try:
    with open(DATA_PATH, "r") as file:
        sequence_data = json.load(file)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {DATA_PATH}")
    exit()

if not sequence_data:
    raise ValueError(f"No data found in {DATA_PATH}")

print(f"Loaded {len(sequence_data)} samples.")

# %%
# --- Process all samples ---
batch_of_token_ids = []

for idx, sample in enumerate(sequence_data):
    sample_prompt = sample.get("prompt")
    sample_generation = sample.get("response")
    
    if not sample_generation:
        print(f"Warning: 'response' not found in sample {idx}, skipping.")
        continue
    if not sample_prompt:
        print(f"Warning: 'prompt' not found in sample {idx}, skipping.")
        continue
    
    # Format the prompt using chat template
    message_prompt = [{"role": "user", "content": sample_prompt}]
    formatted_prompt_tokens = tokenizer.apply_chat_template(
        message_prompt, add_generation_prompt=True, tokenize=True)
    generation_tokens = tokenizer.encode(sample_generation, add_special_tokens=False)
    
    # Combine the formatted prompt and generation tokens
    formatted_tokens = formatted_prompt_tokens + generation_tokens
    
    batch_of_token_ids.append(formatted_tokens)

if not batch_of_token_ids:
    raise ValueError(f"No valid samples with 'prompt' and 'response' found in {DATA_PATH}")

print(f"Processed {len(batch_of_token_ids)} valid token sequences.")

# %%
# --- Define Target Tokens ---
target_token_strs = ["</think>"]
target_token_ids = torch.tensor([tokenizer.encode(
    s, add_special_tokens=False)[0] for s in target_token_strs])
print(
    f"Target tokens: '{target_token_strs}', IDs: {target_token_ids.tolist()}")


# %%
# --- Compute Statistics ---
# Find the position of the target token in each sequence
target_token_positions = []
sequence_lengths = []

for idx, token_ids in enumerate(batch_of_token_ids):
    sequence_lengths.append(len(token_ids))
    
    # Find the first occurrence of any target token
    target_pos = None
    for pos, token_id in enumerate(token_ids):
        if token_id in target_token_ids:
            target_pos = pos
            break
    
    if target_pos is not None:
        target_token_positions.append(target_pos)
    else:
        print(f"Warning: Target token not found in sequence {idx}")

# Compute statistics
if target_token_positions:
    import numpy as np
    
    positions_array = np.array(target_token_positions)
    
    print(f"\n--- Statistics for Target Token Positions ---")
    print(f"Number of sequences with target token: {len(target_token_positions)}")
    print(f"Mean position: {positions_array.mean():.2f}")
    print(f"Median position: {np.median(positions_array):.2f}")
    print(f"Min position: {positions_array.min()}")
    print(f"Max position: {positions_array.max()}")
    print(f"Std deviation: {positions_array.std():.2f}")
    
    # Plot histogram
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(target_token_positions, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Position')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Target Token Positions ({target_token_strs[0]})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_token_position_distribution.png', dpi=300)
    print(f"\nHistogram saved to 'target_token_position_distribution.png'")
    plt.show()
    
    # Plot sequence lengths for context
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Sequence Lengths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sequence_length_distribution.png', dpi=300)
    print(f"Sequence length histogram saved to 'sequence_length_distribution.png'")
    plt.show()
else:
    print("No target tokens found in any sequence.")

# %%
# --- Count samples with length <= 5000 and containing target token ---
max_length_threshold = 5000
valid_samples_count = 0

for idx, seq_len in enumerate(sequence_lengths):
    if seq_len <= max_length_threshold and idx < len(target_token_positions):
        # Check if this sequence has a target token position recorded
        # (target_token_positions is parallel to sequences that had target tokens)
        valid_samples_count += 1

# Alternative: more explicit check
valid_samples_count = 0
for idx in range(len(sequence_lengths)):
    seq_len = sequence_lengths[idx]
    has_target = idx < len(target_token_positions) and target_token_positions[idx] is not None
    
    # Actually, target_token_positions only contains positions for sequences that had the target
    # So we need to track which original indices had targets
    # Let's recount properly:

# Recount with proper tracking
valid_samples_count = 0
for idx, (seq_len, target_pos) in enumerate(zip(sequence_lengths, target_token_positions)):
    if seq_len <= max_length_threshold:
        valid_samples_count += 1

print(f"\n--- Samples with length <= {max_length_threshold} and containing target token ---")
print(f"Count: {valid_samples_count} out of {len(target_token_positions)} sequences with target token")
print(f"Percentage: {100 * valid_samples_count / len(target_token_positions):.2f}%")

# %%
