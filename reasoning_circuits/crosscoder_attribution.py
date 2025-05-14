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
BATCH_SIZE = 8

GENERATION_SEED = 3
INIT_SEED = 42

# DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/outputs/wait_subsequences_from_outputs.json"
DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/data/wait_subsequences_from_data.json"
SAVE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/results"

FT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# CROSSCODER_TYPE = "L1"
CROSSCODER_TYPE = "BatchTopK"
CROSSCODER_LAYER = 23
CROSSCODER_DEVICE = "cuda:0"
CROSSCODER_PATH = f"/disk/u/troitskiid/data/checkpoints/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/cc_weights.pt"

BASELINE = 0

# %%
# --- Load Model and Tokenizer ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

print(f"Loading model {FT_MODEL_NAME}...")
model = LanguageModel(
    FT_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
base_model = LanguageModel(
    BASE_MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(
    FT_MODEL_NAME,
    use_fast=True,
    add_bos_token=True,
)
print("Models and Tokenizer loaded.")

# %%
# --- Load CROSSCODER (from test_crosscoder.py) ---
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
        subsequence_data = json.load(file)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {DATA_PATH}")
    exit()

# Extract token ID sequences
batch_of_token_ids = [item["subsequence_tokens"]
                      for item in subsequence_data if "subsequence_tokens" in item]

if not batch_of_token_ids:
    raise ValueError(f"No 'subsequence_tokens' found in {DATA_PATH}")

print(f"Loaded {len(batch_of_token_ids)} token sequences.")

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
target_token_strs = ["wait", "Wait", " wait", " Wait"]
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
    return log_sum_exp_probs_per_item.mean()

# --- Define CROSSCODER Attribution Computation Function  ---


def compute_crosscoder_attribution(ft_model: LanguageModel, base_model: LanguageModel, baseline: float, crosscoder: CrossCoder, tokens: torch.Tensor, attention_mask: torch.Tensor, metric_fn: Callable, crosscoder_layer_idx: int) -> torch.Tensor:
    """
    Computes crosscoder latent attribution using nnsight by taking gradients w.r.t. crosscoder latents.
    The metric is applied to the last token, but attribution is calculated across all sequence positions.

    Args:
        ft_model: The distilled model.
        base_model: The base model.
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
        f"Starting nnsight trace for layer {crosscoder_layer_idx} ...")
    # Pass both tokens (as input_ids) and attention_mask to trace
    # nnsight passes these kwargs to the model's forward method

    # Get batch size and sequence length for reshaping later

    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]

    with base_model.trace(tokens, kwargs={'attention_mask': attention_mask}):
        layer = base_model.model.layers[crosscoder_layer_idx]
        hidden_state = layer.output[0]  # Shape: (batch, seq, d_model)
        base_hidden_state = hidden_state.save()

    base_hidden_state = base_hidden_state.to(CROSSCODER_DEVICE)

    with ft_model.trace(tokens, kwargs={'attention_mask': attention_mask}):

        layer = ft_model.model.layers[crosscoder_layer_idx]
        hidden_state = layer.output[0]  # Shape: (batch, seq, d_model)
        hidden_state_grad = hidden_state.grad  # Shape: (batch, seq, d_model)

        ft_hidden_state = hidden_state.to(CROSSCODER_DEVICE)
        # Stack along a new dimension (n_models=2)
        # Shape: (batch_size, 2, seq_len, d_model)
        hidden_states_stack = torch.stack(
            [base_hidden_state, ft_hidden_state], dim=1)

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

        # Operates on (batch, seq, d_model)
        hidden_state_grad[~attention_mask] = 0
        # Operates on (batch, seq, d_crosscoder)
        crosscoder_latents[~attention_mask] = 0

        output_logits = model.output.logits
        # Metric calculated on last token logits
        metric_value = ns.apply(metric_fn, output_logits)

        # Approximate the effect of setting the latents to the baseline on the metric
        # Gradient flows back from the metric calculated on the last token,
        # but the attribution is computed across all positions using the gradient at each position.

        # Select the decoder weights for the second model (index 1, assuming 0 is base, 1 is ft)
        decoder_weight_ft = crosscoder.decoder.weight[1]
        # Shape: (d_crosscoder, d_model)

        attribution = (
            einops.einsum(
                decoder_weight_ft,
                hidden_state_grad,
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
        base_model=base_model, ft_model=model, baseline=BASELINE, crosscoder=crosscoder, tokens=batch_tokens, attention_mask=batch_attention_mask,
        metric_fn=metric_fn_prepared, crosscoder_layer_idx=CROSSCODER_LAYER
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
        json_save_path = f"{SAVE_PATH}/{CROSSCODER_TYPE}-Crosscoder/crosscoder_attributions_l{CROSSCODER_LAYER}_mean_last_token_data.json"
        json_save_dir = os.path.dirname(json_save_path)
        os.makedirs(json_save_dir, exist_ok=True)

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
save_path = f"{SAVE_PATH}/{CROSSCODER_TYPE}-Crosscoder/crosscoder_attributions_l{CROSSCODER_LAYER}_data.pt"
save_dir = os.path.dirname(save_path)
os.makedirs(save_dir, exist_ok=True)

print(f"\nSaving combined attributions to {save_path}...")
torch.save(crosscoder_attribution, save_path)
print("Attributions saved.")
# %%
