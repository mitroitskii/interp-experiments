# %%
# --- Imports ---
from typing import TypeVar
import gc
import torch
import os
import json

TT = TypeVar("TT")

# %%
# --- Constants ---
K = 50  # Number of top/bottom latents for the mean attribution

# DATA_SOURCE = "outputs"
DATA_SOURCE = "data"

CROSSCODER_LAYER = 7
CROSSCODER_TYPE = "L1"
# CROSSCODER_TYPE = "BatchTopK"
SAVE_PATH = "."
ATTRIBUTIONS_TENSOR_PATH = f"{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/attributions_{DATA_SOURCE}.pt"

# %%
# --- Load Attributions ---
crosscoder_attribution = torch.load(ATTRIBUTIONS_TENSOR_PATH)
crosscoder_attribution_last_token = crosscoder_attribution[:, -1, :]

print(
    f"Selected last token attribution. Shape: {crosscoder_attribution_last_token.shape}")
del crosscoder_attribution
gc.collect()

print(
    f"Last Token CROSSCODER Attribution shape: {crosscoder_attribution_last_token.shape}")

analysis_tensor = crosscoder_attribution_last_token.cpu(
) if crosscoder_attribution_last_token.numel() > 0 else crosscoder_attribution_last_token

# %%
# --- Analysis for mean over all samples (last token) ---

print(f"\n--- Analysis for mean over all samples (last token) ---")

if analysis_tensor.numel() > 0:
    mean_attribution = analysis_tensor.mean(dim=0)  # Shape: (d_crosscoder)
    num_latents = mean_attribution.shape[0]
    num_to_show = min(K, num_latents)

    if num_to_show > 0:
        # --- Top K (Descending) for mean ---
        top_attr_values, top_attr_indices = torch.topk(
            mean_attribution, num_to_show, largest=True)

        print(
            f"\nTop {num_to_show} attribution latents for mean (last token, descending):")
        top_latents_data = []
        for i in range(num_to_show):
            idx = top_attr_indices[i].item()
            val = top_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            top_latents_data.append({"index": idx, "value": val})

        # Print top indices in the requested format
        top_indices_list = top_attr_indices.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_mean_top_{num_to_show}_descending = {top_indices_list}")

        # --- Bottom K (Ascending) for mean ---
        bottom_attr_values, bottom_attr_indices = torch.topk(
            mean_attribution, num_to_show, largest=False)

        print(
            f"\nBottom {num_to_show} attribution latents for mean (last token, ascending):")
        bottom_latents_data = []
        for i in range(num_to_show):
            idx = bottom_attr_indices[i].item()
            val = bottom_attr_values[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            bottom_latents_data.append({"index": idx, "value": val})

        # Print bottom indices in the requested format
        bottom_indices_list = bottom_attr_indices.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_mean_bottom_{num_to_show}_ascending = {bottom_indices_list}")

        # --- Save to JSON for mean ---
        json_save_path = f"{SAVE_PATH}/{CROSSCODER_TYPE}-Crosscoder/L{CROSSCODER_LAYER}R/attributions_mean_last_token_{DATA_SOURCE}.json"
        json_save_dir = os.path.dirname(json_save_path)
        os.makedirs(json_save_dir, exist_ok=True)

        data_to_save = {
            "top": top_latents_data,
            "bottom": bottom_latents_data
        }

        with open(json_save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(
            f"\nSaved mean top/bottom latents with values to {json_save_path}")
        # --- End of added section ---

    else:
        print(
            f"\nNot enough latents to show top/bottom k.")
else:
    print("\nSkipping mean attribution analysis as no data was processed.")

# %%
