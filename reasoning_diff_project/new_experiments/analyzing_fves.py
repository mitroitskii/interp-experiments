# %%
from dictionary_learning.dictionary import BatchTopKCrossCoder
from torch.nn.functional import cosine_similarity
import torch as th
import numpy as np
import nnsight
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import warnings
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
th.set_grad_enabled(False)
from scipy.interpolate import make_interp_spline
import seaborn as sns
import csv
import sys
sys.path.append("..")

# %%
# crosscoder_path = "/share/u/models/crosscoder_checkpoints/DeepScaleR_vs_Qwen2.5-Math_L15/ae.pt"
# crosscoder = "L15"
# extra_args = []
# exp_name = "eval_crosscoder"
# exp_id = ""
# base_layer = 0
# reasoning_layer = 1

# model_device = "cuda"
# coder_device = "cuda"
# calc_device = "cpu"
# coder = BatchTopKCrossCoder.from_pretrained(crosscoder_path)
# coder = coder.to(coder_device)
# num_layers, activation_dim, dict_size = coder.encoder.weight.shape

# base_model = nnsight.LanguageModel("Qwen/Qwen2.5-Math-1.5B", device_map=model_device)
# ft_model = nnsight.LanguageModel("agentica-org/DeepScaleR-1.5B-Preview",device_map=model_device)
# csv_path = Path("csv_files/kl_divergence_results_with_percents.csv")


# layer = 15
# dataset = load_dataset("koyena/OpenR1-Math-220k-formatted")['train']
# dataset = dataset.take(1000)

# For QWEN
BOS = 151646
USER = 151644
ASSISTANT = 151645
NEWLINE = 198
THINK_START = 151648
THINK_END = 151649
EOS = 151643
#%%
def calculate_fve(activations, reconstructed_activations):
    # Calculate fraction of variance explained.
    total_variance = th.sum(activations ** 2)
    residual_variance = th.sum((activations - reconstructed_activations) ** 2)
    fve = 1 - residual_variance / total_variance
    return fve

def calculate_fve_full(csv_path="csv_files/fve_results.csv"):
    counter = 0
    # Use a list of dictionaries to store results per example
    results = []
    for row in tqdm(dataset):
        # if counter == 25:
        #     continue
        print(f"EXAMPLE NO.{counter}", flush=True)
        print("================================")
        text = row["message_in_chat_template"]
        # print(text)
        print("================================")
        tokenized_text = ft_model.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text['input_ids']
        # print(input_ids)
        positions_think_start = [i for i, tid in enumerate(input_ids[0]) if tid == THINK_START]
        positions_think_end = [i for i, tid in enumerate(input_ids[0]) if tid == THINK_END]
        # get activations to give to crosscoder
        # get output to measure kl divergence
        # base model
        with base_model.trace(input_ids) as _:
            base_activations = base_model.model.layers[layer].output.save()
        # get activations to give to crosscoder
        # get output to measure kl divergence
        # ft model
        with ft_model.trace(input_ids) as _:
            ft_activations = ft_model.model.layers[layer].output.save()

        # stack activations retrieved
        activations = th.stack([base_activations[0].to(coder_device), ft_activations[0].to(coder_device)], dim=1)
        # remove the first dimension of size 1 since coder doesn't like that
        activations = activations.squeeze()
        activations = einops.rearrange(activations, 'l t h -> t l h', l=activations.shape[0], t=activations.shape[1], h=activations.shape[2]).to(coder_device)
        # run crosscoder
        crosscoder_activations = coder(activations.float())
        # split crosscoder activations to respective way.
        base_coder = crosscoder_activations[:,0,:].unsqueeze(0).to(coder_device)
        ft_coder =  crosscoder_activations[:, 1, :].unsqueeze(0).to(coder_device)
        print(base_activations[0].shape)
        print(base_coder.shape)
        fve_base_think_start = calculate_fve(base_activations[0][:,positions_think_start,:], base_coder[:,positions_think_start,:])
        fve_ft_think_start = calculate_fve(ft_activations[0][:,positions_think_start,:], ft_coder[:,positions_think_start,:])
        fve_base_think_end = calculate_fve(base_activations[0][:,positions_think_end,:], base_coder[:,positions_think_end,:])
        fve_ft_think_end = calculate_fve(ft_activations[0][:,positions_think_end,:], ft_coder[:,positions_think_end,:])
        print(fve_base_think_start, fve_ft_think_start, fve_base_think_end, fve_ft_think_end)
        
        results.append({
            "fve_base_think_start": fve_base_think_start.item(),
            "fve_ft_think_start": fve_ft_think_start.item(),
            "fve_base_think_end": fve_base_think_end.item(),
            "fve_ft_think_end": fve_ft_think_end.item()
        })
        counter += 1
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(results[0].keys())
        for result in results:
            writer.writerow(result.values())


# %%
    
if __name__ == "__main__":
    #main()
    #calculate_fve_full()
    df = pd.read_csv("../csv_files/fve_results.csv")
    print(df.head())
    # plot the fve of the base model and the ft model for the think start and think end positions
    plt.figure(figsize=(10, 5))
    x_values = df.index # Use the DataFrame index for the x-axis
    plt.scatter(x_values, df["fve_base_think_start"], label="Base Model Think Start", s=10) # s controls the marker size
    plt.scatter(x_values, df["fve_ft_think_start"], label="FT Model Think Start", s=10)
    plt.scatter(x_values, df["fve_base_think_end"], label="Base Model Think End", s=10)
    plt.scatter(x_values, df["fve_ft_think_end"], label="FT Model Think End", s=10)
    plt.xlabel("Example Number") # Add an x-axis label
    plt.ylabel("FVE") # Add a y-axis label
    plt.title("FVE Comparison at Think Start/End Tokens") # Add a title
    plt.legend()
    plt.savefig("../plots/fve_scatter_plot.png") # Changed filename to avoid overwriting
    plt.show()