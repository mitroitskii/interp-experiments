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
from sae_lens import SAE

TT = TypeVar("TT")

# %%
# --- Constants ---
BATCH_SIZE = 16

GENERATION_SEED = 3
INIT_SEED = 42

DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/outputs/wait_subsequences.json"
SAVE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/results"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

SAE_LAYER = 23
SAE_DEVICE = "cuda:3"
RELEASE = "llama_scope_r1_distill"
SAE_ID = f"l{SAE_LAYER}r_400m_slimpajama_400m_openr1_math"

BASELINE = 0.0

# %%
# --- Seed ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

# %%
# --- Load Model and Tokenizer ---
print(f"Loading model {MODEL_NAME}...")
model = LanguageModel(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_bos_token=True,
)
print("Model and Tokenizer loaded.")

# %%
# --- Load SAE (from test_sae.py) ---
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)
print(f"Loading SAE {SAE_ID} for layer {SAE_LAYER}...")
sae = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device=SAE_DEVICE,
)[0]
sae.to(dtype=torch.bfloat16)
sae.eval()  # Ensure SAE is in evaluation mode
print("SAE loaded.")

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

# --- Define SAE Attribution Computation Function  ---

def compute_sae_attribution(model: LanguageModel, sae: SAE, tokens: torch.Tensor, attention_mask: torch.Tensor, metric_fn: Callable, sae_layer_idx: int, baseline: float) -> torch.Tensor:
    """
    Computes SAE latent attribution using nnsight by taking gradients w.r.t. SAE latents.
    The metric is applied to the last token, but attribution is calculated across all sequence positions.

    Args:
        model: The nnsight LanguageModel.
        sae: The loaded SAE model.
        tokens: Input tokens tensor (input_ids).
        attention_mask: Attention mask tensor (shape: batch, seq).
        metric_fn: The metric function to compute gradients against (applied to last token).
        sae_layer_idx: The index of the layer where the SAE is applied.
        baseline: The baseline value for the attribution calculation.

    Returns:
        torch.Tensor: Attribution scores for each SAE latent feature at each
                      token position, shape (batch_size, seq_len, d_sae).
    """
    print(
        f"Starting nnsight trace for layer {sae_layer_idx} ...")
    # Pass both tokens (as input_ids) and attention_mask to trace
    # nnsight passes these kwargs to the model's forward method
    with model.trace(tokens, kwargs={'attention_mask': attention_mask}):
        layer = model.model.layers[sae_layer_idx]
        hidden_state = layer.output[0]  # Shape: (batch, seq, d_model)
        hidden_state_grad = hidden_state.grad  # Shape: (batch, seq, d_model)

        # Ensure hidden state is on the correct device for SAE
        hidden_state_on_sae_device = hidden_state.to(sae.device)
        sae_latents = ns.apply(
            sae.encode, hidden_state_on_sae_device).save()  # Shape: (batch, seq, d_sae)

        # Zero out the gradients and latents
        hidden_state_grad[~attention_mask] = 0
        sae_latents[~attention_mask] = 0

        output_logits = model.output.logits
        # Metric calculated on last token logits
        metric_value = ns.apply(metric_fn, output_logits).save()

        # Approximate the effect of setting the latents to the baseline on the metric
        # Gradient flows back from the metric calculated on the last token,
        # but the attribution is computed across all positions using the gradient at each position.
        attribution = (
            einops.einsum(
                sae.W_dec,  # (d_sae, d_model)
                hidden_state_grad,  # (batch, seq, d_model)
                "d_sae d_model, batch seq d_model -> batch seq d_sae",
            )
            * (sae_latents - 
            )  # (batch, seq, d_sae)
        ).save()  # Final shape: (batch, seq, d_sae)

        # Backpropagate to compute the gradients of the metric w.r.t. the latents
        metric_value.backward()
        # - call .backward() at the end of the trace to let nnsight build the computation graph

    print("NNsight trace completed.")
    return attribution


# --- Prepare Metric ---
metric_fn_prepared = partial(mean_multi_target_log_prob_metric,
                             target_token_ids=target_token_ids)


# %%
# --- Batched Processing ---
num_sequences = input_tokens.shape[0]
all_sae_attributions = []
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

    batch_sae_attribution = compute_sae_attribution(
        model=model, sae=sae, tokens=batch_tokens, attention_mask=batch_attention_mask,
        metric_fn=metric_fn_prepared, sae_layer_idx=SAE_LAYER, baseline=BASELINE
    )

    print(f"Batch {i // BATCH_SIZE + 1} completed.\n")
    all_sae_attributions.append(batch_sae_attribution.cpu())
    total_processed_sequences += current_batch_size

    print("Deleting intermediate tensors...\n")
    del batch_sae_attribution
    del batch_tokens, batch_attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Crucial for managing memory in loops
    gc.collect()

print("\nBatched processing finished. Combining results...")

# --- Combine Batch Results ---
if all_sae_attributions:
    sae_attribution = torch.cat(all_sae_attributions, dim=0)
    print("Results combined.")
    # Select only the attribution for the last token position
    # Shape: (total_sequences, d_sae)
    sae_attribution_last_token = sae_attribution[:, -1, :]
    print(
        f"Selected last token attribution. Shape: {sae_attribution_last_token.shape}")
else:
    print("No batches were processed.")
    sae_attribution_last_token = torch.empty(0)

del all_sae_attributions
gc.collect()

# %%
# --- Results & Analysis ---
# Use the last token attribution for analysis
print(f"\n--- Results (Last Token Attributions) ---")
print(f"Last Token SAE Attribution shape: {sae_attribution_last_token.shape}")

# Analyze top attributing latents (summed/averaged over batch dimension only)
analysis_tensor = sae_attribution_last_token.cpu(
) if sae_attribution_last_token.numel() > 0 else sae_attribution_last_token

if analysis_tensor.numel() > 0:
    # Sum attribution over the batch dimension
    total_attribution_per_latent = analysis_tensor.sum(dim=0)  # Shape: (d_sae)
    k = 50  # Number of top/bottom latents to find
    num_latents = total_attribution_per_latent.shape[0]
    num_to_show = min(k, num_latents)

    if num_to_show > 0:
        # --- Top K (Descending) ---
        top_attr_values, top_attr_indices = torch.topk(
            total_attribution_per_latent, num_to_show, largest=True)

        print(
            f"\nTop {num_to_show} attribution latents (summed over batch for last token, descending):")
        for i in range(num_to_show):
            idx = top_attr_indices[i].item()
            val = top_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")

        # Print top indices in the requested format
        top_indices_list = top_attr_indices.cpu().tolist()
        print(f"\nl{SAE_LAYER}_top_{num_to_show}_descending = {top_indices_list}")

        # --- Bottom K (Ascending) ---
        bottom_attr_values, bottom_attr_indices = torch.topk(
            total_attribution_per_latent, num_to_show, largest=False)

        print(
            f"\nBottom {num_to_show} attribution latents (summed over batch for last token, ascending):")
        # torch.topk with largest=False returns the k smallest elements, sorted ascendingly
        for i in range(num_to_show):
            idx = bottom_attr_indices[i].item()
            val = bottom_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")

        # Print bottom indices in the requested format
        bottom_indices_list = bottom_attr_indices.cpu().tolist()
        print(f"\nl{SAE_LAYER}_bottom_{num_to_show}_ascending = {bottom_indices_list}")

    else:
        print("\nNot enough latents to show top/bottom k.")
else:
    print("\nSkipping attribution analysis as no data was processed.")


# %%
# --- Save Results ---
# Save the last token attribution
save_path = f"{SAVE_PATH}/sae_attributions_zero_baseline_l{SAE_LAYER}.pt"
save_dir = os.path.dirname(save_path)
os.makedirs(save_dir, exist_ok=True)

print(f"\nSaving combined attributions to {save_path}...")
torch.save(sae_attribution, save_path)
print("Attributions saved.")
# %%
