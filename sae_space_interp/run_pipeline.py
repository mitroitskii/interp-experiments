# Reworked from https://github.com/safety-research/open-source-em-features/
# %% Imports
import os
import torch
from pathlib import Path
from typing import Dict, List, Any
# from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_quick_list
from neuronpedia_utils import get_neuronpedia_quick_list_url
from sae_loading import load_sae_from_path
from sae_decomposition import decompose_difference_vectors_on_sae

# %% Constants
# Directory where vector files are stored
VECTORS_DIR = Path("./vectors")

# Output file for results
OUTPUT_FILE = "results.txt"

# SAE configuration
LAYER = 12
RELEASE = "llama_scope_lxr_32x"
SAE_ID = f"l{LAYER}r_32x"
TOP_K = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mapping of steering vector ategories
VECTOR_CATEGORIES = {
    "idx0": "Recalling Mathematical Formulas",
    "idx1": "Retrieving Factual Knowledge",
    "idx2": "Listing considerations",
    "idx3": "Conditional Outcome Projection",
    "idx4": "Conditional Causal Reasoning",
    "idx5": "Drawing Conclusions",
    "idx6": "Restating Given Data",
    "idx7": "Verifying Intermediate Steps",
    "idx8": "Confirming Reasoning Steps",
    "idx9": "Mathematical Computation Steps",
    "idx10": "Numeric Calculation Steps",
    "idx11": "Task Formulation",
    "idx12": "Proposing Possibilities",
    "idx13": "Planning Next Steps",
    "idx14": "Expressing Uncertainty"
}

# %% Load SAE
print(f"Loading SAE for layer {LAYER}...")
sae = load_sae_from_path(release=RELEASE, sae_id=SAE_ID, device=DEVICE)
print(f"SAE loaded successfully.")

# %% Process Vectors and Generate Neuronpedia Links
print("\nProcessing vectors and generating interpretation links...")

results_links = []

for idx, category in VECTOR_CATEGORIES.items():

    vector_path = VECTORS_DIR / f"llama-3.1-8b_{idx}.pt"
    
    if not vector_path.exists():
        print(f"Skipping {idx} ({category}): File not found at {vector_path}")
        continue
    
    print(f"\nAnalyzing: {category} ({idx})")
    
    # Load vector
    vec_data = torch.load(vector_path, map_location=DEVICE)
    if isinstance(vec_data, dict):
        vec = vec_data[idx]
    else:
        vec = vec_data
    
    try:

        sae_result = decompose_difference_vectors_on_sae(
            vector=vec,
            sae=sae,
            layer=LAYER,
            device=DEVICE,
            top_k=TOP_K
        )
        
        top_indices = sae_result['top_features']['indices']
        top_similarities = sae_result['top_features']['similarities']
    
        try:
            link = get_neuronpedia_quick_list_url(sae, top_indices, list_name=category)
            # link = get_neuronpedia_quick_list(sae, top_indices, name=category)
            results_links.append(f"{idx} ({category}):\n  Link: {link}\n  Top Indices: {top_indices}\n  Similarities: {[round(s, 4) for s in top_similarities]}\n")
        except Exception as e:
            print(f"Error generating Neuronpedia link for {category}: {e}")
            results_links.append(f"{idx} ({category}):\n  Error generating link: {e}\n  Top Indices: {top_indices}\n  Similarities: {[round(s, 4) for s in top_similarities]}\n")

    except Exception as e:
        print(f"Error during decomposition for {category}: {e}")

# %% Save Results
print(f"\nSaving results to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    for line in results_links:
        f.write(line + "\n")

print("Done!")
