import torch
import torch.nn.functional as F
from functools import partial
from transformer_lens import HookedTransformer, ActivationCache
from typing import Callable, TypeVar
import einops

TT = TypeVar("TT")

model = HookedTransformer.from_pretrained("gpt2-small")
model.set_use_attn_result(True)

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
clean_tokens = model.to_tokens(prompts)

# 1. Define the TWO target tokens
target_token_strs = ["Wait", "wait"]
# Ensure target_token_ids is a tensor for easier indexing later
target_token_ids = torch.tensor([model.to_single_token(
    s) for s in target_token_strs], device=model.cfg.device)
print(
    f"Target tokens: '{target_token_strs}', IDs: {target_token_ids.tolist()}")

# 2. Create the new metric function


def mean_multi_target_log_prob_metric(logits: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean log probability of a set of target tokens
    at the final sequence position, averaged across the batch.
    """
    # Get logits for the last token position
    last_logits = logits[:, -1, :]  # Shape: [batch, d_vocab]

    # Convert logits to log probabilities
    log_probs = F.log_softmax(last_logits, dim=-1)  # Shape: [batch, d_vocab]

    # Gather the log probabilities for ALL specified target tokens for each batch item
    # Indexing with a tensor extracts the columns corresponding to the token IDs
    # Shape: [batch, num_targets]
    multi_target_log_probs = log_probs[:, target_token_ids]

    # Compute the log sum of probabilities (ie the cumulative probability of either token occuring)
    log_sum_exp_probs_per_item = torch.logsumexp(
        multi_target_log_probs, dim=1)  # Shape: [batch]

    # Return the mean across the batch
    return log_sum_exp_probs_per_item.mean()


def filter_not_qkv_input(name): return "_input" not in name


def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")
    value = metric(model(tokens))  # Metric calculation happens here
    value.backward()  # Gradients calculated based on the metric's value
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


# 3. Prepare the metric function for the call (binding the target_token_id)
metric_fn = partial(mean_multi_target_log_prob_metric,
                    target_token_ids=target_token_ids)
# Alternative: metric_fn = lambda logits: target_log_prob_metric(logits, target_token_id=target_token_id)

# Calculate value, activations, and gradients using the new metric
# Note: We no longer need the ioi_metric, get_logit_diff, CLEAN_BASELINE,
# CORRUPTED_BASELINE for this specific gradient calculation.
clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, metric_fn  # Pass the new metric function
)
print(
    f"Clean Value (Mean Log Prob of '{target_token_strs}'): {clean_value:.4f}")
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))

# Optionally, calculate for corrupted tokens if you want to compare values
# corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
#     model, corrupted_tokens, metric_fn
# )
# print(f"Corrupted Value (Mean Log Prob of '{target_token_str}'): {corrupted_value:.4f}")


def create_attention_attr(
    clean_cache: ActivationCache,
    clean_grad_cache: ActivationCache,
    threshold: float  # Add threshold parameter
) -> TT["batch", "layer", "head_index", "dest", "src"]:
    """
    Calculates the linear approximation of the change in the metric
    when setting attention scores to a specific threshold.

    Args:
        clean_cache: ActivationCache containing forward pass activations.
        clean_grad_cache: ActivationCache containing backward pass gradients.
        threshold: The threshold value T to approximate setting activations to.

    Returns:
        A tensor representing the approximated change in the metric for each
        attention link, shape: ["batch", "layer", "head_index", "dest", "src"]
    """
    attention_stack = torch.stack(
        [clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    attention_grad_stack = torch.stack(
        [clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    )
    # Approximation of the change in the metric when setting attention scores to a specific threshold
    attention_attr_approx_change = attention_grad_stack * \
        (threshold - attention_stack)
    attention_attr_approx_change = einops.rearrange(
        attention_attr_approx_change,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr_approx_change


# --- Attribution Calculation ---
# Define the threshold value
threshold_value = 0.0  # Example: Approximates setting attention scores to 0.0

# The attribution calculation now uses gradients derived from the log prob metric
# and incorporates the threshold
attention_attr = create_attention_attr(
    clean_cache, clean_grad_cache, threshold_value)  # Pass threshold
# Update print statement
print(
    f"Calculated attention attribution with respect to log prob metric and threshold={threshold_value}.")
print("Attention Attr shape:", attention_attr.shape)
