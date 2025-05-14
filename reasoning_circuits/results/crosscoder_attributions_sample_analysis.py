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
SAMPLE_INDEX = 16
K = 50  # Number of top/bottom latents for the specific sample

CROSSCODER_LAYER = 7
CROSSCODER_TYPE = "BatchTopK"
# CROSSCODER_TYPE = "L1"
SAVE_PATH = "."
ATTRIBUTIONS_TENSOR_PATH = f"{CROSSCODER_TYPE}-Crosscoder/crosscoder_attributions_l{CROSSCODER_LAYER}_data.pt"

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
# --- Analysis for a specific entry (e.g., sample 16) ---

print(f"\n--- Analysis for sample {SAMPLE_INDEX} ---")

if analysis_tensor.numel() > 0 and analysis_tensor.shape[0] > SAMPLE_INDEX:
    attribution_sample = analysis_tensor[SAMPLE_INDEX]  # Shape: (d_crosscoder)
    num_latents_sample = attribution_sample.shape[0]
    num_to_show_sample = min(K, num_latents_sample)

    if num_to_show_sample > 0:
        # --- Top K (Descending) for sample ---
        top_attr_values_sample, top_attr_indices_sample = torch.topk(
            attribution_sample, num_to_show_sample, largest=True)

        print(
            f"\nTop {num_to_show_sample} attribution latents for sample {SAMPLE_INDEX} (last token, descending):")
        top_latents_data_sample = []
        for i in range(num_to_show_sample):
            idx = top_attr_indices_sample[i].item()
            val = top_attr_values_sample[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            top_latents_data_sample.append({"index": idx, "value": val})

        # Print top indices in the requested format
        top_indices_list_sample = top_attr_indices_sample.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_sample{SAMPLE_INDEX}_top_{num_to_show_sample}_descending = {top_indices_list_sample}")

        # --- Bottom K (Ascending) for sample ---
        bottom_attr_values_sample, bottom_attr_indices_sample = torch.topk(
            attribution_sample, num_to_show_sample, largest=False)

        print(
            f"\nBottom {num_to_show_sample} attribution latents for sample {SAMPLE_INDEX} (last token, ascending):")
        bottom_latents_data_sample = []
        # torch.topk with largest=False returns the k smallest elements, sorted ascendingly
        for i in range(num_to_show_sample):
            idx = bottom_attr_indices_sample[i].item()
            val = bottom_attr_values_sample[i].item()
            print(f"  Latent {idx}: Attribution = {val:.4f}")
            bottom_latents_data_sample.append({"index": idx, "value": val})

        # Print bottom indices in the requested format
        bottom_indices_list_sample = bottom_attr_indices_sample.cpu().tolist()
        print(
            f"\nl{CROSSCODER_LAYER}_sample{SAMPLE_INDEX}_bottom_{num_to_show_sample}_ascending = {bottom_indices_list_sample}")

        # --- Save to JSON for sample ---
        json_save_path = f"{SAVE_PATH}/{CROSSCODER_TYPE}-Crosscoder/crosscoder_attributions_l{CROSSCODER_LAYER}_sample{SAMPLE_INDEX}_last_token_data.json"
        json_save_dir = os.path.dirname(json_save_path)
        os.makedirs(json_save_dir, exist_ok=True)

        data_to_save = {
            "top": top_latents_data_sample,
            "bottom": bottom_latents_data_sample
        }

        with open(json_save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(
            f"\nSaved sample {SAMPLE_INDEX} top/bottom latents with values to {json_save_path}")
        # --- End of added section ---

    else:
        print(
            f"\nNot enough latents in sample {SAMPLE_INDEX} to show top/bottom k.")
elif analysis_tensor.numel() == 0:
    print("\nSkipping sample-specific attribution analysis as no data was processed.")
else:
    print(
        f"\nSkipping sample-specific attribution analysis as sample {SAMPLE_INDEX} does not exist (batch size: {analysis_tensor.shape[0]}).")

# %%
