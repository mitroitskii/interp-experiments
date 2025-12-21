# %%
# --- Imports ---
from typing import Callable, TypeVar
from functools import partial
import os
import json
import random
import gc
import argparse
import einops
import torch
import torch.nn.functional as F
import nnsight as ns
from nnsight import LanguageModel
from transformers import AutoTokenizer, set_seed
from dictionary_learning import BatchTopKCrossCoder, CrossCoder

TT = TypeVar("TT")

# %%
# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="CrossCoder Attribution Analysis")
parser.add_argument("--layer", type=int, required=True, help="Layer number for crosscoder")
args = parser.parse_args()

# %%
# --- Constants ---
BATCH_SIZE = 3

GENERATION_SEED = 3
INIT_SEED = 42

DATA_SOURCE = "outputs" # FIXME remove?
# DATA_SOURCE = "data"

DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/outputs/new_generated_outputs.json" # prompt + response containing the full response with closing </think> token
SAVE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/results"

# --- Models Setup ---
REASONING_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# Model to perform attribution on
ATTRIBUTION_MODEL = "Reasoning"
# ATTRIBUTION_MODEL = "Base" 

TOKENIZER_MODEL_ID = REASONING_MODEL_ID

# --- Crosscoder Setup ---
CROSSCODER_TYPE = "L1"
# CROSSCODER_TYPE = "BatchTopK"

CROSSCODER_LAYER = args.layer
CROSSCODER_PATH = f"/disk/u/troitskiid/data/checkpoints/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/cc_weights.pt"

CROSSCODER_DEVICE = "cuda:7"

# --- Attribution Setup ---
BASELINE = 0

# %%
# --- Load Model and Tokenizer ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

print(
    f"Loading base model: {BASE_MODEL_ID}...")
base_model = LanguageModel(
    BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
print(f"Loading reasoning model: {REASONING_MODEL_ID}...")
reasoning_model = LanguageModel(
    REASONING_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)

# Short name for the attribution model, used in save paths
if ATTRIBUTION_MODEL == "Reasoning":
    print(f"Using reasoning model for attribution: {ATTRIBUTION_MODEL}...")
    attribution_model = reasoning_model
elif ATTRIBUTION_MODEL == "Base":
    print(f"Using base model for attribution: {ATTRIBUTION_MODEL}...")
    attribution_model = base_model

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

# Note: 432 out of 495 sequences with target token are <= 5000 tokens long
# see output_stats.py for calculations

# Filter sequences to only include those with length <= 5000 tokens
MAX_SEQUENCE_LENGTH = 5000
filtered_batch_of_token_ids = []

for idx, token_ids in enumerate(batch_of_token_ids):
    if len(token_ids) <= MAX_SEQUENCE_LENGTH:
        filtered_batch_of_token_ids.append(token_ids)
    else:
        print(f"Filtering out sequence {idx} with length {len(token_ids)} > {MAX_SEQUENCE_LENGTH}")

batch_of_token_ids = filtered_batch_of_token_ids

if not batch_of_token_ids:
    raise ValueError(f"No sequences with length <= {MAX_SEQUENCE_LENGTH} tokens found.")

print(f"After filtering: {len(batch_of_token_ids)} sequences with length <= {MAX_SEQUENCE_LENGTH} tokens.")



# %%
# --- Pad and batch token ID sequences ---
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad token is set
# Prepare input for tokenizer.pad
# Each element in the list should be a list of token IDs
encoded_inputs = {'input_ids': batch_of_token_ids}

# Pad the batch to the length of the longest sequence in the batch
padded_batch = tokenizer.pad(
    encoded_inputs,
    padding='longest',  # Pad to the length of the longest sequence
    return_tensors="pt"
)

input_tokens = padded_batch['input_ids']
attention_mask = padded_batch['attention_mask']


print(f"Data loaded and sequences padded. Input shape: {input_tokens.shape}")
print(f"Attention mask shape: {attention_mask.shape}")


# %%
# --- Decode the first sequence for verification ---
if input_tokens.shape[0] > 0:
    decoded_first_input = tokenizer.decode(
        input_tokens[0], skip_special_tokens=False)
    print(
        f"Decoded first input sequence (with padding): {decoded_first_input}\n")
else:
    print("No input sequences to decode.")

# %%
# --- Define Target Tokens ---
target_token_strs = ["</think>"]
target_token_ids = torch.tensor([tokenizer.encode(
    s, add_special_tokens=False)[0] for s in target_token_strs])
print(
    f"Target tokens: '{target_token_strs}', IDs: {target_token_ids.tolist()}")

# %%
# --- Define Metric Function ---


def mean_multi_target_log_prob_metric(logits: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean log probability of a set of target tokens
    at the final sequence position, averaged across the batch.
    """
    # Ensure target_token_ids is on the same device as logits
    target_token_ids = target_token_ids.to(logits.device)
    last_logits = logits[:, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)
    multi_target_log_probs = log_probs[:, target_token_ids]
    log_sum_exp_probs_per_item = torch.logsumexp(
        multi_target_log_probs, dim=-1)
    # mean here leads to the same results as sum in terms of relative ordering of attribution scores ->
    # using mean simply scales all gradients by 1/batch_size compared to sum, preserving relative feature importance
    return log_sum_exp_probs_per_item.mean()

# --- Define CROSSCODER Attribution Computation Function  ---


def compute_crosscoder_attribution(attribution_model: LanguageModel, base_model: LanguageModel, reasoning_model: LanguageModel, baseline: float, crosscoder: CrossCoder, tokens: torch.Tensor, attention_mask: torch.Tensor, metric_fn: Callable, crosscoder_layer_idx: int) -> torch.Tensor:
    """
    Computes crosscoder latent attribution using nnsight by taking gradients w.r.t. crosscoder latents.
    The metric is applied to the last token, but attribution is calculated across all sequence positions.

    Args:
        attribution_model: The model on which attribution is performed and whose output is used for the metric.
        base_model: The base model (always first input to crosscoder).
        reasoning_model: The reasoning model (always second input to crosscoder).
        crosscoder: The loaded crosscoder model.
        tokens: Input tokens tensor (input_ids).
        attention_mask: Attention mask tensor (shape: batch, seq).
        metric_fn: The metric function to compute gradients against (applied to last token).
        crosscoder_layer_idx: The index of the layer where the crosscoder is applied.
        baseline: The baseline value for the attribution calculation.

    Returns:
        torch.Tensor: Attribution scores for each crosscoder latent feature at each
                      token position, shape (batch_size, seq_len, d_crosscoder).
    """
    print(
        f"Starting nnsight trace for layer {crosscoder_layer_idx})...")
    # Pass both tokens (as input_ids) and attention_mask to trace
    # nnsight passes these kwargs to the model's forward method

    # Get batch size and sequence length for reshaping later

    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]

    # Determine which model is the reference model (the one NOT being attributed)
    # The crosscoder stack is always [base_model, reasoning_model]
    if attribution_model == reasoning_model:
        reference_model = base_model
        attribution_model_idx = 1  # reasoning model is second in stack
    else:  # attribution_model == base_model
        reference_model = reasoning_model
        attribution_model_idx = 0  # base model is first in stack

    with reference_model.trace(tokens, kwargs={'attention_mask': attention_mask}):
        layer_ref = reference_model.model.layers[crosscoder_layer_idx]
        hidden_state_ref = layer_ref.output[0]  # Shape: (batch, seq, d_model)
        reference_model_hidden_state_for_crosscoder = hidden_state_ref.save()

    reference_model_hidden_state_for_crosscoder = reference_model_hidden_state_for_crosscoder.to(
        CROSSCODER_DEVICE)

    with attribution_model.trace(tokens, kwargs={'attention_mask': attention_mask}):

        layer_attr = attribution_model.model.layers[crosscoder_layer_idx]
        # Shape: (batch, seq, d_model)
        hidden_state_attr = layer_attr.output[0]
        # Shape: (batch, seq, d_model)
        hidden_state_attr_grad = hidden_state_attr.grad

        attribution_model_hidden_state_for_crosscoder = hidden_state_attr.to(
            CROSSCODER_DEVICE)

        # Stack always in order: base_model first, reasoning_model second
        # Shape: (batch_size, 2, seq_len, d_model)
        if attribution_model == reasoning_model:
            # reference is base, attribution is reasoning
            hidden_states_stack = torch.stack(
                [reference_model_hidden_state_for_crosscoder, attribution_model_hidden_state_for_crosscoder], dim=1)
        else:
            # attribution is base, reference is reasoning
            hidden_states_stack = torch.stack(
                [attribution_model_hidden_state_for_crosscoder, reference_model_hidden_state_for_crosscoder], dim=1)

        # Reshape for crosscoder.encode: Combine batch and sequence dimensions
        # Shape: (batch_size * seq_len, 2, d_model)
        hidden_states_rearranged = einops.rearrange(
            hidden_states_stack, 'batch n_models seq d_model -> (batch seq) n_models d_model')

        # Apply crosscoder.encode - Assuming output shape (batch * seq, d_crosscoder)
        crosscoder_latents_flat = ns.apply(
            crosscoder.encode, hidden_states_rearranged)

        # Reshape latents back to (batch, seq, d_crosscoder)
        crosscoder_latents = einops.rearrange(
            crosscoder_latents_flat, '(batch seq) d_crosscoder -> batch seq d_crosscoder', batch=batch_size, seq=seq_len)

        # Convert attention_mask to boolean and add feature dimension for broadcasting
        # attention_mask shape: (batch, seq) -> (batch, seq, 1)
        attention_mask_bool = attention_mask.bool().unsqueeze(-1)

        # Apply masking by multiplying (this zeros out padded positions)
        # Operates on (batch, seq, d_model) * (batch, seq, 1) -> (batch, seq, d_model)
        hidden_state_attr_grad = hidden_state_attr_grad * attention_mask_bool
        # Operates on (batch, seq, d_crosscoder) * (batch, seq, 1) -> (batch, seq, d_crosscoder)
        crosscoder_latents = crosscoder_latents * attention_mask_bool

        # Use logits from the attribution_model
        output_logits = attribution_model.output.logits
        # Metric calculated on last token logits
        metric_value = ns.apply(metric_fn, output_logits)

        # Approximate the effect of setting the latents to the baseline on the metric
        # Gradient flows back from the metric calculated on the last token,
        # but the attribution is computed across all positions using the gradient at each position.

        # Select the decoder weights for the attribution model (using attribution_model_idx)
        decoder_weight_for_attribution_model = crosscoder.decoder.weight[attribution_model_idx]
        # Shape: (d_crosscoder, d_model)

        attribution = (
            einops.einsum(
                decoder_weight_for_attribution_model,
                hidden_state_attr_grad,  # Gradient from the attribution_model's hidden state
                "d_crosscoder d_model, batch seq d_model -> batch seq d_crosscoder",
            )
            * (crosscoder_latents - baseline)  # (batch, seq, d_crosscoder)
        ).save()

        # Backpropagate to compute the gradients of the metric w.r.t. the latents
        metric_value.backward()
        # - call .backward() at the end of the trace to let nnsight build the computation graph

    print("NNsight trace completed.")
    return attribution


# % %%
# --- Prepare Metric ---
metric_fn_prepared = partial(mean_multi_target_log_prob_metric,
                             target_token_ids=target_token_ids)


# %%
# --- Batched Processing ---
num_sequences = input_tokens.shape[0]
all_crosscoder_attributions = []
total_processed_sequences = 0

print(f"Starting batched processing with batch size {BATCH_SIZE}...")

for i in range(0, num_sequences, BATCH_SIZE):
    batch_start = i
    batch_end = min(i + BATCH_SIZE, num_sequences)
    print(
        f"Processing batch {i // BATCH_SIZE + 1}/{(num_sequences + BATCH_SIZE - 1) // BATCH_SIZE}...")

    batch_tokens = input_tokens[batch_start:batch_end]
    batch_attention_mask = attention_mask[batch_start:batch_end]
    current_batch_size = batch_tokens.shape[0]

    random.seed(GENERATION_SEED)
    torch.manual_seed(GENERATION_SEED)
    set_seed(GENERATION_SEED)

    batch_crosscoder_attribution = compute_crosscoder_attribution(
        attribution_model=attribution_model,
        base_model=base_model,
        reasoning_model=reasoning_model,
        baseline=BASELINE,
        crosscoder=crosscoder,
        tokens=batch_tokens,
        attention_mask=batch_attention_mask,
        metric_fn=metric_fn_prepared,
        crosscoder_layer_idx=CROSSCODER_LAYER
    )

    print(f"Batch {i // BATCH_SIZE + 1} completed.\n")
    all_crosscoder_attributions.append(batch_crosscoder_attribution.cpu())
    total_processed_sequences += current_batch_size

    print("Deleting intermediate tensors...\n")
    del batch_crosscoder_attribution
    del batch_tokens, batch_attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Crucial for managing memory in loops
    gc.collect()

print("\nBatched processing finished. Combining results...")

# --- Combine Batch Results ---
if all_crosscoder_attributions:
    crosscoder_attribution = torch.cat(all_crosscoder_attributions, dim=0)
    print("Results combined.")
    # Select only the attribution for the last token position
    # Shape: (total_sequences, d_crosscoder)
    crosscoder_attribution_last_token = crosscoder_attribution[:, -1, :]
    print(
        f"Selected last token attribution. Shape: {crosscoder_attribution_last_token.shape}")
else:
    print("No batches were processed.")
    crosscoder_attribution_last_token = torch.empty(0)

del all_crosscoder_attributions
gc.collect()


# %%
# --- Results & Analysis ---
# Use the last token attribution for analysis
print(f"\n--- Results (Last Token Attributions) ---")
print(
    f"Last Token CROSSCODER Attribution shape: {crosscoder_attribution_last_token.shape}")

# Define base directory for saving results, including model and layer
model_specific_results_dir = f"{SAVE_PATH}/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/{ATTRIBUTION_MODEL}"
os.makedirs(model_specific_results_dir, exist_ok=True)
print(f"Results will be saved in: {model_specific_results_dir}")

# Define the prefix for attribution result files (JSON and PT)
results_file_prefix = f"{model_specific_results_dir}/attributions"

# Analyze top attributing latents (summed/averaged over batch dimension only)
analysis_tensor = crosscoder_attribution_last_token.cpu(
) if crosscoder_attribution_last_token.numel() > 0 else crosscoder_attribution_last_token

if analysis_tensor.numel() > 0:
    # Sum attribution over the batch dimension
    total_attribution_per_latent = analysis_tensor.mean(
        dim=0)  # Shape: (d_crosscoder)
    k = 50  # Number of top/bottom latents to find
    num_latents = total_attribution_per_latent.shape[0]
    num_to_show = min(k, num_latents)

    if num_to_show > 0:
        # --- Top K (Descending) ---
        top_attr_values, top_attr_indices = torch.topk(
            total_attribution_per_latent, num_to_show, largest=True)

        print(
            f"\nTop {num_to_show} attribution latents (summed over batch for last token, descending):")
        top_latents_data = []
        for i in range(num_to_show):
            idx = top_attr_indices[i].item()
            val = top_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            top_latents_data.append({"index": idx, "value": val})

        # Print top indices in the requested format
        top_indices_list = top_attr_indices.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_top_{num_to_show}_descending_indices = {top_indices_list}")

        # --- Bottom K (Ascending) ---
        bottom_attr_values, bottom_attr_indices = torch.topk(
            total_attribution_per_latent, num_to_show, largest=False)

        print(
            f"\nBottom {num_to_show} attribution latents (summed over batch for last token, ascending):")
        bottom_latents_data = []
        # torch.topk with largest=False returns the k smallest elements, sorted ascendingly
        for i in range(num_to_show):
            idx = bottom_attr_indices[i].item()
            val = bottom_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            bottom_latents_data.append({"index": idx, "value": val})

        # Print bottom indices in the requested format
        bottom_indices_list = bottom_attr_indices.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_bottom_{num_to_show}_ascending_indices = {bottom_indices_list}")

        # --- Save to JSON ---
        json_save_path = f"{results_file_prefix}_mean_last_token_{DATA_SOURCE}.json"
        # json_save_dir = os.path.dirname(json_save_path) # Already created by model_specific_results_dir
        # os.makedirs(json_save_dir, exist_ok=True)

        data_to_save = {
            "top": top_latents_data,
            "bottom": bottom_latents_data
        }

        with open(json_save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nSaved top/bottom latents with values to {json_save_path}")

    else:
        print("\nNot enough latents to show top/bottom k.")
else:
    print("\nSkipping attribution analysis as no data was processed.")


# %%
# --- Save Results ---
# Save the last token attribution

# save_path = f"{SAVE_PATH}/crosscoder_attributions_l{CROSSCODER_LAYER}_outputs.pt"
pt_save_path = f"{results_file_prefix}_{DATA_SOURCE}.pt"
# pt_save_dir = os.path.dirname(pt_save_path) # Already created by model_specific_results_dir
# os.makedirs(pt_save_dir, exist_ok=True)

print(f"\nSaving combined attributions to {pt_save_path}...")
torch.save(crosscoder_attribution, pt_save_path)
print("Attributions saved.")
# %%
