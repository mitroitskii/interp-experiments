# %%
from dictionary_learning.dictionary_learning.dictionary import BatchTopKCrossCoder
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
crosscoder_path = "/share/u/models/crosscoder_checkpoints/DeepScaleR_vs_Qwen2.5-Math_L15/ae.pt"
crosscoder = "L15"
extra_args = []
exp_name = "eval_crosscoder"
exp_id = ""
base_layer = 0
reasoning_layer = 1

model_device = "auto"
coder_device = "cuda:6"
calc_device = "cpu"
coder = BatchTopKCrossCoder.from_pretrained(crosscoder_path)
coder = coder.to(coder_device)
num_layers, activation_dim, dict_size = coder.encoder.weight.shape

base_model = nnsight.LanguageModel("Qwen/Qwen2.5-Math-1.5B", device_map=model_device)
ft_model = nnsight.LanguageModel("agentica-org/DeepScaleR-1.5B-Preview",device_map=model_device)


layer = 15
dataset = load_dataset("koyena/OpenR1-Math-220k-formatted")['train']
dataset = dataset.take(1000)

# For QWEN
BOS = 151646
USER = 151644
ASSISTANT = 151645
NEWLINE = 198
THINK_START = 151648
THINK_END = 151649
EOS = 151643

# %%
def calculate_kl_divergence(logits_p, logits_q, temperature=1.0, constant=0, token_wise=True):
    """
    Calculate KL divergence between two distributions represented by logits
    
    Args:
        logits_p: Output logits from the first model (shape: batch_size x vocab_size)
        logits_q: Output logits from the second model (shape: batch_size x vocab_size)
        temperature: Temperature for softening the distributions
        
    Returns:
        KL divergence (scalar)
    """
    # Apply temperature scaling
    p_logits = logits_p / temperature
    q_logits = logits_q / temperature
    
    # Convert logits to probabilities
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    if token_wise:
        # Calculate KL divergence element-wise: p(x) * (log p(x) - log q(x))
    # The input to F.kl_div should be log probabilities (log q)
    # The target should be probabilities (p) when log_target=False
    # Add constant to avoid log(0) issues
    # Shape: batch_size x seq_len x vocab_size
        kl_elementwise = F.kl_div(
            input=th.log(q_probs + constant), # log q_probs
            target=p_probs,                 # p_probs
            reduction='none',               # Compute element-wise KL term
            log_target=False
        )    
        # Sum over the vocabulary dimension to get KL divergence per token position
        # KL(P || Q) = sum_{vocab} p(x) * log(p(x)/q(x))
        # Shape: batch_size x seq_len
        kl_div_per_token = th.sum(kl_elementwise, dim=-1)
        # Calculate the mean over the batch and sequence dimensions
        # mean_kl_div = th.mean(kl_div_per_token)
        
        return kl_div_per_token
    else:
    # Calculate KL divergence: p(x) * log(p(x)/q(x))
        kl_div = F.kl_div(
            input=th.log(q_probs + constant),
            target=p_probs,
            reduction='batchmean',
            log_target=False
        )
        return kl_div
# %%
def kl_divergence_test(token_wise=True, csv_path=Path("csv_files/full_kl_divergence_results_token_wise.csv")):
    counter = 0
    # Use a list of dictionaries to store results per example
    results = []
    for row in tqdm(dataset):
        print(f"EXAMPLE NO.{counter}", flush=True)
        print("================================")
        text = row["message_in_chat_template"]
        tokenized_text = ft_model.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text['input_ids']
        
        # get activations to give to crosscoder
        # get output to measure kl divergence
        # base model
        with base_model.trace(input_ids) as _:
            base_activations = base_model.model.layers[layer].output.save()
            base_output = base_model.lm_head.output.save()

        # get activations to give to crosscoder
        # get output to measure kl divergence
        # ft model
        with ft_model.trace(input_ids) as _:
            ft_activations = ft_model.model.layers[layer].output.save()
            ft_output = ft_model.lm_head.output.save()

        # stack activations retrieved
        activations = th.stack([base_activations[0].to(coder_device), ft_activations[0].to(coder_device)], dim=1)
        # remove the first dimension of size 1 since coder doesn't like that
        activations = activations.squeeze()
        activations = einops.rearrange(activations, 'l t h -> t l h', l=activations.shape[0], t=activations.shape[1], h=activations.shape[2]).to(coder_device)
        # run crosscoder
        crosscoder_activations = coder(activations.float())
        # split crosscoder activations to respective way.
        base_coder = crosscoder_activations[:,0,:].unsqueeze(0)
        ft_coder =  crosscoder_activations[:, 1, :].unsqueeze(0)

        with base_model.edit() as base_model_edited:
            # have to edit it this way since the coder gives out [1536] but there are 159 tokens.
            base_model.model.layers[layer].output[0][:] = base_coder

        with base_model_edited.trace(input_ids) as tracer:
            base_intervened_output = base_model_edited.lm_head.output.save()

        with ft_model.edit() as ft_model_edited:
            ft_model.model.layers[layer].output[0][:] = ft_coder

        with ft_model_edited.trace(input_ids) as tracer:
            ft_intervened_output = ft_model_edited.lm_head.output.save()

        del ft_model_edited
        del base_model_edited
        del base_activations, ft_activations, activations, crosscoder_activations, base_coder, ft_coder
        base_output = base_output.to(calc_device)
        ft_output = ft_output.to(calc_device)   
        base_intervened_output = base_intervened_output.to(calc_device)
        ft_intervened_output = ft_intervened_output.to(calc_device)    
        
        # Inside the loop, before KL calculations
        #print(f"FT output stats: min={th.min(ft_output)}, max={th.max(ft_output)}, has_nan={th.isnan(ft_output).any()}, has_inf={th.isinf(ft_output).any()}", flush=True)
        kl_base_base = calculate_kl_divergence(base_output, base_output, token_wise=token_wise) # Should be near 0
        kl_base_ft = calculate_kl_divergence(base_output, ft_output, token_wise=token_wise) # ORIGINAL
        kl_ft_ft = calculate_kl_divergence(ft_output, ft_output, token_wise=token_wise) # Should be near 0
        kl_base_base_intervened = calculate_kl_divergence(base_output, base_intervened_output, token_wise=token_wise)
        kl_ft_ft_intervened = calculate_kl_divergence(ft_output, ft_intervened_output, token_wise=token_wise)
        del base_output, ft_output

        kl_base_ft_intervened = calculate_kl_divergence(base_intervened_output, ft_intervened_output, token_wise=token_wise) # INTERVENED

        del base_intervened_output, ft_intervened_output        
        kl_act_diff = kl_base_ft - kl_base_ft_intervened # Difference between original and intervened KL
        kl_act_diff_percent = (kl_act_diff / kl_base_ft) * 100

        # Store results for the current example
        # average per prompt / example
        for i in range(len(input_ids[0])):
            current_results = {
                "example_id": counter,
                "encoded_token": input_ids[0][i].item(),
                "kl_base_ft": kl_base_ft[0][i].item(),
                "kl_base_base": kl_base_base[0][i].item(),
                "kl_ft_ft": kl_ft_ft[0][i].item(),
                "kl_base_ft_intervened": kl_base_ft_intervened[0][i].item(),
                "kl_base_base_intervened": kl_base_base_intervened[0][i].item(),
                "kl_ft_ft_intervened": kl_ft_ft_intervened[0][i].item(),
                "kl_act_diff": kl_act_diff[0][i].item(),
                "kl_act_diff_percent": kl_act_diff_percent[0][i].item(),
            }
            results.append(current_results)

        # Clean up tensors to free GPU memory
        
        
        # Delete KL divergence tensors explicitly
        del kl_base_ft, kl_base_base, kl_ft_ft, kl_base_ft_intervened
        del kl_base_base_intervened, kl_ft_ft_intervened, kl_act_diff
        del current_results # Not strictly necessary, but good practice

        # print("================================")
        gc.collect()
        th.cuda.empty_cache()
        counter += 1
        # Check if we need to write the batch to CSV (every 10 examples)
        if (counter + 1) % 10 == 0:
            if results: # Only write if there are results in the batch
                print(f"Writing batch ending at example {counter} to CSV...", flush=True)
                df_batch = pd.DataFrame(results)

                # Append to CSV
                write_header = not csv_path.exists()
                df_batch.to_csv(csv_path, mode='a', header=write_header, index=False)

                # Clear the batch list
                results = []
                print(f"Batch written. Cleared results list.", flush=True)
                del df_batch
                gc.collect()
                th.cuda.empty_cache()
        # counter += 1
# %%
    


def plot_kl_act_diff_percent_distribution(df):
    # Ensure the index is suitable for plotting (numeric)
    plt.figure(figsize=(10, 6))

    # Create the histogram plot showing raw counts
    # Removed kde=True to show only the histogram bars
    sns.histplot(data=df, x='kl_act_diff_percent') 

    # Add labels and title
    plt.xlabel("KL Activation Difference (%)")
    plt.ylabel("Number of Examples (Count)") # Y-axis shows raw counts
    plt.title("Histogram of KL Activation Difference Percentage (Raw Counts)") # Updated title
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    save_path = 'plots/kl_act_diff_percent_histogram_token_wise.png' # Updated filename
    # Ensure plots directory exists if needed
    # import os
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path) 
    print(f"Saved raw count histogram to {save_path}")

    plt.show()
    plt.close() # Close the figure

def plot_kl_scatter_comparison(df):
    """
    Generates a scatter plot comparing KL(Base || FT) vs KL(FT || FT Intervened).
    Uses transparency and small markers to handle potentially large datasets.

    Args:
        df (pd.DataFrame): DataFrame containing KL divergence results with columns
                           'kl_base_ft' and 'kl_ft_ft_intervened'.
    """
    plt.figure(figsize=(10, 10)) # Square figure is often good for scatter plots

    # Define the columns to plot
    x_col = 'kl_base_ft'
    y_col = 'kl_ft_ft_intervened'

    # Create the scatter plot
    # Use small markers (s) and transparency (alpha) for dense data
    plt.scatter(df[x_col], df[y_col], alpha=0.3, s=5, label='Per-Example KL Divergence')

    # Add an identity line (y=x) for reference
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x line')

    # Add labels and title
    plt.xlabel(f"{x_col} (KL Divergence)")
    plt.ylabel(f"{y_col} (KL Divergence)")
    plt.title(f"Scatter Plot Comparison: {y_col} vs {x_col}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes equal scale if desired
    plt.tight_layout()

    # Save the figure
    plt.savefig('plots/kl_scatter_comparison.png')
    plt.show()

def get_top_k_examples(df, k, tokenizer):
    # Sort the DataFrame by kl_act_diff_percent in descending order
    sorted_df = df.sort_values(by='kl_act_diff_percent', ascending=False)
    
    # Get the top k examples
    top_k_examples = sorted_df.head(k)
    # Decode the tokens
    # Ensure the tokenizer is accessible (it's defined globally as base_model.tokenizer)
    try:
        top_k_examples['decoded_token'] = top_k_examples['encoded_token'].apply(
            lambda token_id: tokenizer.decode([token_id], skip_special_tokens=False)
        )
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        top_k_examples['decoded_token'] = top_k_examples['encoded_token'].astype(str) # Fallback

    # Plot the top k examples
    plt.figure(figsize=(10, 10))
    plt.scatter(top_k_examples['kl_base_ft'], top_k_examples['kl_ft_ft_intervened'], alpha=0.3, s=5, label='Per-Example KL Divergence')
    plt.xlabel("KL(Base || FT)")
    plt.ylabel("KL(FT || FT Intervened)")
    plt.title(f"Top {k} Examples by KL Activation Difference Percentage")
    plt.legend()
    # save the plot
    plt.savefig(f'plots/top_{k}_kl_comparison.png')
    plt.show()
    plt.close()

    return top_k_examples

# Example usage (assuming df is loaded elsewhere):
# plot_kl_scatter_comparison(your_dataframe)

# ... existing code ...
# You might want to rename or remove the old plot_kl_comparison_histograms function
# def plot_kl_comparison_histograms(df):
# # Ensure the index is suitable for plotting (numeric)
#     # Assuming 'df' is your DataFrame containing the results.
#     plt.figure(figsize=(12, 7))
#
#     # Create histograms for the four columns
#     # Using element='step' and fill=False makes comparing multiple histograms easier
#     # sns.histplot(data=df, x='kl_base_base', element='step', fill=False, linewidth=1.5, label='KL(Base || Base)', color='blue')
#     # sns.histplot(data=df, x='kl_ft_ft', element='step', fill=False, linewidth=1.5, label='KL(FT || FT)', color='green')
#     sns.histplot(data=df, x='kl_base_ft', element='step', fill=False, linewidth=1.5, label='KL(Base || FT)', color='red', linestyle='--')
#     sns.histplot(data=df, x='kl_ft_ft_intervened', element='step', fill=False, linewidth=1.5, label='KL(FT || FT Replaced with Crosscoder)', color='purple', linestyle=':')
#
#     # Add labels and title
#     plt.xlabel("KL Divergence Value")
#     plt.ylabel("Number of Examples (Count)") # y-axis now shows counts
#     plt.title("Histogram Comparison of KL Divergence Values")
#     plt.legend() # Add a legend to distinguish the lines
#     plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#
#     # Save the figure
#     plt.savefig('plots/kl_comparison_histograms.png')
#
#     plt.show()




if __name__ == "__main__":
    #kl_divergence_test(token_wise=True)
    #main()
    df = pd.read_csv("csv_files/kl_divergence_results_token_wise.csv")
    # Filter the DataFrame
    df = df[df['example_id'] <= 1000]
    plot_kl_scatter_comparison(df)
    plot_kl_act_diff_percent_distribution(df)
    # plot_kl_act_diff_percent_distribution(df)
    # plot_kl_comparison_histograms(df)
    # k = top 25% of the dataset    
    k = int(len(df) * 0.25)
    get_top_k_examples(df, k=k, tokenizer=ft_model.tokenizer) # Example call with k=20