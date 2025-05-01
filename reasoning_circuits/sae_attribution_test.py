# %%
# --- Imports ---
from typing import Callable, TypeVar
from functools import partial
import json
import random
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from transformers import AutoTokenizer, set_seed
from sae_lens import SAE

TT = TypeVar("TT")

# %%
# --- Constants ---
GENERATION_SEED = 3
INIT_SEED = 42

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

SAE_LAYER = 15
RELEASE = "llama_scope_r1_distill"
SAE_ID = f"l{SAE_LAYER}r_400m_slimpajama_400m_openr1_math"
SAE_DEVICE = "cuda" if torch.cuda.is_available(
) else "cpu"

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
    MODEL_NAME, device_map="auto")
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
    device=SAE_DEVICE
)[0]
sae.eval()  # Ensure SAE is in evaluation mode
print("SAE loaded.")

# %%
# --- Load Data ---
print("Loading data...")
with open("outputs/interesting_outputs.json", "r") as file:
    interesting_outputs = json.load(file)

input_strings = []
for item in interesting_outputs:
    for output_item in item["outputs"]:
        input_strings.append(output_item["output"])

if not input_strings:
    raise ValueError("No input strings found in interesting_outputs.json")

print(f"Loaded {len(input_strings)} input strings.")

# %%
# Tokenize the input strings as a batch
tokenizer.pad_token = tokenizer.eos_token
tokenized = tokenizer(
    input_strings,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=model.config.max_position_embeddings,
    return_tensors="pt"
)
input_tokens = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

print(f"Data loaded and tokenized. Input shape: {input_tokens.shape}\n")
decoded_first_input = tokenizer.decode(
    input_tokens[0], skip_special_tokens=False)
print(f"Attention mask shape: {attention_mask.shape}\n")
print(
    f"Decoded first input (with padding/truncation): {decoded_first_input}\n")

# %%
# --- Define Target Tokens ---
target_token_strs = ["Wait", "wait"]
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

# --- Define SAE Attribution Computation Function (Direct Latent Gradient Method) ---`  `


def compute_sae_attribution(model: LanguageModel, sae: SAE, tokens: torch.Tensor, attention_mask: torch.Tensor, metric_fn: Callable, sae_layer_idx: int, baseline: float):
    """
    Computes SAE latent attribution using nnsight by taking gradients directly w.r.t. latents.

    Args:
        model: The nnsight LanguageModel.
        sae: The loaded SAE model.
        tokens: Input tokens tensor (input_ids).
        attention_mask: Attention mask tensor.
        metric_fn: The metric function to compute gradients against.
        sae_layer_idx: The index of the layer where the SAE is applied.
        baseline: The baseline value for the attribution calculation.

    Returns:
        Tuple: (attribution, sae_latents, sae_latents_grad, metric_value)
    """
    print(
        f"Starting nnsight trace for layer {sae_layer_idx} (direct latent gradient method)...")
    # Pass both tokens (as input_ids) and attention_mask to trace
    # nnsight passes these kwargs to the model's forward method
    with model.trace(tokens, kwargs={'attention_mask': attention_mask}):
        # 1. Get hidden state and replace with SAE reconstruction
        layer = model.model.layers[sae_layer_idx]
        # hidden_state is the output of the layer's forward pass *before* intervention
        hidden_state = layer.output[0]

        # Ensure hidden_state is on the SAE device if necessary
        hidden_state_device = hidden_state.device
        sae = sae.to(hidden_state_device)

        # Shape: [batch, seq, d_sae]
        sae_latents = sae.encode(hidden_state)

        # This ensures gradients flow *through* the SAE operation correctly
        # Decoder needs latents as input
        sae_reconstruction = sae.decode(sae_latents)
        # Ensure reconstruction is on the original hidden_state device
        layer.output[0][:] = sae_reconstruction.to(hidden_state_device)

        # 2. Calculate metric based on the *modified* computation graph
        output_logits = model.output.logits
        # Save the scalar metric value
        # metric_fn now handles device transfer for target_ids
        metric_value = metric_fn(output_logits)

        # 3. Trigger backward pass to compute gradients based on the metric
        metric_value.backward()

        # 4. Get gradients of SAE latents w.r.t the metric
        # This gradient now reflects the sensitivity *after* the intervention
        # Shape: [batch, seq, d_sae]
        sae_latents_grad = sae_latents.grad

        # 5. Calculate attribution: grad * (baseline - activation)
        # Ensure baseline is broadcastable or has the same shape as sae_latents
        attribution = sae_latents_grad * (baseline - sae_latents)

    print("nnsight trace completed.")
    return (
        attribution.save(),
        sae_latents.save(),
        sae_latents_grad.save(),
        metric_value.save()
    )


# --- Prepare Metric and Compute Attribution ---
metric_fn_prepared = partial(mean_multi_target_log_prob_metric,
                             target_token_ids=target_token_ids)  # Pass CPU tensor here

# %%
# Use the direct gradient function
random.seed(GENERATION_SEED)
torch.manual_seed(GENERATION_SEED)
set_seed(GENERATION_SEED)

sae_attribution, sae_latents_values, sae_latents_grads, final_metric_value = compute_sae_attribution(
    model=model,
    sae=sae,
    tokens=input_tokens,
    attention_mask=attention_mask,  # Pass the attention mask here
    metric_fn=metric_fn_prepared,
    sae_layer_idx=SAE_LAYER,
    baseline=BASELINE
)

# %%
print(f"\n--- Results (Direct Latent Gradient Method) ---")
print(
    f"Final Metric Value (Mean Log Prob of '{target_token_strs}'): {final_metric_value.item():.4f}")
print(f"SAE Latents shape: {sae_latents_values.shape}")
print(f"SAE Latents Gradients shape: {sae_latents_grads.shape}")
print(f"SAE Attribution shape: {sae_attribution.shape}")
print(
    f"Calculated SAE latent attribution using direct latent gradients and baseline={BASELINE}.")

# Analyze top attributing latents ---
total_attribution_per_latent = sae_attribution.sum(
    dim=(0, 1))  # Shape: [d_sae]
top_k = 10
top_attr_values, top_attr_indices = torch.topk(
    total_attribution_per_latent.abs(), top_k)

print(f"\nTop {top_k} absolute attribution latents (summed over batch/seq):")
for i in range(top_k):
    idx = top_attr_indices[i].item()
    val = total_attribution_per_latent[idx].item()  # Get original signed value
    print(f"  Latent {idx}: Attribution = {val:.4f}")

# --- Mean Attribution of Latents ---
mean_attribution_per_latent = sae_attribution.mean(dim=(0, 1))
print(f"\nMean Attribution of Latents: {mean_attribution_per_latent}")


# %%
