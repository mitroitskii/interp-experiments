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
SAMPLE_INDEX = 184  # Process a single sample

GENERATION_SEED = 3
INIT_SEED = 42

# DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/outputs/wait_subsequences_from_outputs.json"
DATA_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/data/wait_subsequences_from_data.json"
# SAVE_PATH = "/disk/u/troitskiid/projects/interp-experiments/reasoning_circuits/results" # No longer saving files this way

FT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

# CROSSCODER_TYPE = "L1"
CROSSCODER_TYPE = "BatchTopK"
CROSSCODER_LAYER = 7
CROSSCODER_DEVICE = "cuda:0"
CROSSCODER_PATH = f"/disk/u/troitskiid/data/checkpoints/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/cc_weights.pt"

# BASELINE = 0 # Baseline removed from attribution calculation as per lean code request

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
        subsequence_data = json.load(file)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {DATA_PATH}")
    exit()

if not subsequence_data or len(subsequence_data) <= SAMPLE_INDEX:
    raise ValueError(
        f"Not enough data in {DATA_PATH} or sample index {SAMPLE_INDEX} is out of bounds."
    )

# Extract token ID sequence for the specified sample
sample_token_ids = subsequence_data[SAMPLE_INDEX].get("subsequence_tokens")

if not sample_token_ids:
    raise ValueError(
        f"'subsequence_tokens' not found in sample {SAMPLE_INDEX} from {DATA_PATH}"
    )

print(f"Loaded token sequence for sample {SAMPLE_INDEX}.")

# %%
# --- Prepare Input Tensor ---
# Convert to tensor and add batch dimension
input_tokens = torch.tensor([sample_token_ids], dtype=torch.long)
print(f"Input shape: {input_tokens.shape}")


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
    # Extract log probabilities for each target token ID
    # This indexes the log_probs tensor (shape [batch_size, vocab_size])
    # with target_token_ids to get log probs for just our target tokens
    # Result shape: [batch_size, num_target_tokens]
    multi_target_log_probs = log_probs[:, target_token_ids]
    # Calculate the log of the sum of exponentials of log probabilities across target tokens
    # This effectively computes log(p(token1) + p(token2) + ...) for each batch item
    # Using logsumexp for numerical stability instead of exponentiating and summing directly
    # dim=-1 means we sum across the target tokens dimension, resulting in one value per batch item
    log_sum_exp_probs_per_item = torch.logsumexp(
        multi_target_log_probs, dim=-1)
    # Return the mean of log sum exp probabilities across the batch
    # This gives us the average log probability of seeing any of our target tokens
    # in the final position, which is a measure of how likely the model is to
    # generate one of our target tokens next
    return log_sum_exp_probs_per_item.mean()

# --- Define CROSSCODER Attribution Computation Function  ---


def compute_crosscoder_outputs(ft_model: LanguageModel, base_model: LanguageModel, crosscoder: CrossCoder, tokens: torch.Tensor, metric_fn: Callable, crosscoder_layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes crosscoder latent attribution and other specified tensors using nnsight.
    The metric is applied to the last token, but attribution is calculated across all sequence positions.

    Args:
        ft_model: The distilled model.
        base_model: The base model.
        crosscoder: The loaded crosscoder model.
        tokens: Input tokens tensor (input_ids). Shape (1, seq_len) for single sample.
        metric_fn: The metric function to compute gradients against (applied to last token).
        crosscoder_layer_idx: The index of the layer where the crosscoder is applied.

    Returns:
        A tuple containing:
        - attribution (torch.Tensor): Attribution scores.
        - hidden_state_grad_saved (torch.Tensor): Saved gradients of the hidden state.
        - crosscoder_latents_saved (torch.Tensor): Saved crosscoder latents.
        - projected_grad (torch.Tensor): Projected gradients.
    """
    print(
        f"Starting nnsight trace for layer {crosscoder_layer_idx} ...")

    # Get batch size and sequence length for reshaping later

    batch_size = tokens.shape[0]
    seq_len = tokens.shape[1]

    with base_model.trace(tokens):
        layer = base_model.model.layers[crosscoder_layer_idx]
        hidden_state = layer.output[0]  # Shape: (batch, seq, d_model)
        base_hidden_state = hidden_state.save()

    base_hidden_state = base_hidden_state.to(CROSSCODER_DEVICE)

    with ft_model.trace(tokens):

        layer = ft_model.model.layers[crosscoder_layer_idx]
        hidden_state = layer.output[0]  # Shape: (batch, seq, d_model)
        hidden_state_grad = hidden_state.grad.save()  # Shape: (batch, seq, d_model)

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
        crosscoder_latents = crosscoder_latents.save()

        output_logits = model.output.logits
        # Metric calculated on last token logits
        metric_value = ns.apply(metric_fn, output_logits)

        # Approximate the effect of setting the latents to the baseline on the metric
        # Gradient flows back from the metric calculated on the last token,
        # but the attribution is computed across all positions using the gradient at each position.

        # Select the decoder weights for the second model (index 1, assuming 0 is base, 1 is ft)
        decoder_weight_ft = crosscoder.decoder.weight[1]
        # Shape: (d_crosscoder, d_model)

        projected_grad = (
            einops.einsum(
                decoder_weight_ft,
                hidden_state_grad,
                "d_crosscoder d_model, batch seq d_model -> batch seq d_crosscoder",
            )
        ).save()

        # Element-wise product with latents
        attribution = (projected_grad * crosscoder_latents).save()

        # Backpropagate to compute the gradients of the metric w.r.t. the latents
        metric_value.backward()
        # - call .backward() at the end of the trace to let nnsight build the computation graph

    print("NNsight trace completed.")
    return attribution, hidden_state_grad, crosscoder_latents, projected_grad


# % %%
# --- Prepare Metric ---
metric_fn_prepared = partial(mean_multi_target_log_prob_metric,
                             target_token_ids=target_token_ids)


# %%
# --- Process Single Sample ---
print(f"Processing sample {SAMPLE_INDEX}...")

random.seed(GENERATION_SEED)
torch.manual_seed(GENERATION_SEED)
set_seed(GENERATION_SEED)

# Ensure tensors are on the correct device for the model
# Models are on 'auto', crosscoder on CROSSCODER_DEVICE.
# nnsight handles internal device placements during trace.
# For Llama models with device_map='auto', inputs typically start on CPU.

attribution, hs_grad, cc_latents, proj_grad = compute_crosscoder_outputs(
    base_model=base_model, ft_model=model, crosscoder=crosscoder,
    tokens=input_tokens,
    metric_fn=metric_fn_prepared, crosscoder_layer_idx=CROSSCODER_LAYER
)

print("Processing completed.\n")

# %%
# --- Results ---
print(f"\n--- Results for Sample {SAMPLE_INDEX} (Last Token Position) ---")

# Ensure tensors are on CPU for analysis if not already
attribution_cpu = attribution.cpu()
hs_grad_cpu = hs_grad.cpu()
cc_latents_cpu = cc_latents.cpu()
proj_grad_cpu = proj_grad.cpu()

# Last token analysis
# For tensors with shape (batch, seq, dim), select last token: tensor[:, -1, :]

print(f"Attribution shape: {attribution_cpu.shape}")
if attribution_cpu.ndim == 3:  # batch, seq, d_crosscoder
    attr_last_token = attribution_cpu[0, -1, :]
    print(
        f"  Non-zero values in last token attribution: {torch.count_nonzero(attr_last_token)}")
# Potentially (seq, d_crosscoder) if batch was squeezed by user
elif attribution_cpu.ndim == 2:
    attr_last_token = attribution_cpu[-1, :]
    print(
        f"  Non-zero values in last token attribution: {torch.count_nonzero(attr_last_token)}")
else:  # Should not happen if batch dim is preserved
    print(
        f"  Attribution tensor has unexpected ndim: {attribution_cpu.ndim}. Full tensor non-zero: {torch.count_nonzero(attribution_cpu)}")


print(f"Hidden State Gradient shape: {hs_grad_cpu.shape}")
if hs_grad_cpu.ndim == 3:  # batch, seq, d_model
    hs_grad_last_token = hs_grad_cpu[0, -1, :]
    print(
        f"  Non-zero values in last token hidden state gradient: {torch.count_nonzero(hs_grad_last_token)}")
else:
    print(
        f"  Hidden state grad tensor has unexpected ndim: {hs_grad_cpu.ndim}. Full tensor non-zero: {torch.count_nonzero(hs_grad_cpu)}")


print(f"CrossCoder Latents shape: {cc_latents_cpu.shape}")
if cc_latents_cpu.ndim == 3:  # batch, seq, d_crosscoder
    cc_latents_last_token = cc_latents_cpu[0, -1, :]
    print(
        f"  Non-zero values in last token CrossCoder latents: {torch.count_nonzero(cc_latents_last_token)}")
else:
    print(
        f"  Crosscoder latents tensor has unexpected ndim: {cc_latents_cpu.ndim}. Full tensor non-zero: {torch.count_nonzero(cc_latents_cpu)}")


print(f"Projected Gradient shape: {proj_grad_cpu.shape}")
if proj_grad_cpu.ndim == 3:  # batch, seq, d_crosscoder
    proj_grad_last_token = proj_grad_cpu[0, -1, :]
    print(
        f"  Non-zero values in last token projected gradient: {torch.count_nonzero(proj_grad_last_token)}")
else:
    print(
        f"  Projected grad tensor has unexpected ndim: {proj_grad_cpu.ndim}. Full tensor non-zero: {torch.count_nonzero(proj_grad_cpu)}")


# Clean up
del attribution, hs_grad, cc_latents, proj_grad
del attribution_cpu, hs_grad_cpu, cc_latents_cpu, proj_grad_cpu
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

print("\nScript finished.")
# %%
