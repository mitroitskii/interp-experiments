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
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import glob
from scipy.stats import entropy  # Import entropy for KL divergence calculation
from scipy.stats import ttest_ind

# %%
# crosscoder_path = "/share/u/models/crosscoder_checkpoints/DeepScaleR_vs_Qwen2.5-Math_L15/ae.pt"
# crosscoder = "L15"
# extra_args = []
# exp_name = "eval_crosscoder"
# exp_id = ""
# base_layer = 0
# reasoning_layer = 1

model_device = "auto"
# coder_device = "cuda:6"
# calc_device = "cpu"
# coder = BatchTopKCrossCoder.from_pretrained(crosscoder_path)
# coder = coder.to(coder_device)
# num_layers, activation_dim, dict_size = coder.encoder.weight.shape

# base_model = nnsight.LanguageModel("Qwen/Qwen2.5-Math-1.5B", device_map=model_device)
ft_model = nnsight.LanguageModel("agentica-org/DeepScaleR-1.5B-Preview",device_map=model_device)


# layer = 15
# dataset = load_dataset("koyena/OpenR1-Math-220k-formatted")['train']
# dataset = dataset.take(1000)

# # For QWEN
# BOS = 151646
# USER = 151644
# ASSISTANT = 151645
# NEWLINE = 198
# THINK_START = 151648
# THINK_END = 151649
# EOS = 151643

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

def plot_kl_scatter_comparison(df, column_name_1='kl_base_ft', column_name_2='kl_base_ft_intervened'):
    """
    Generates a scatter plot comparing KL(Base || FT) vs KL(FT || FT Intervened).
    Uses transparency and small markers to handle potentially large datasets.

    Args:
        df (pd.DataFrame): DataFrame containing KL divergence results with columns
                           'kl_base_ft' and 'kl_ft_ft_intervened'.
    """
    plt.figure(figsize=(10, 10)) # Square figure is often good for scatter plots

    # Define the columns to plot
    x_col = column_name_1
    y_col = column_name_2

    # Create the scatter plot
    # Use small markers (s) and transparency (alpha) for dense data
    plt.scatter(df[x_col], df[y_col], alpha=0.1, s=5, label='Per-Example KL Divergence')

    # Add an identity line (y=x) for reference
    # Dynamic limits based on actual plot data
    lims = [
        np.min([df[x_col].min(), df[y_col].min()]),
        np.max([df[x_col].max(), df[y_col].max()])
    ]
    # Add some padding to limits
    padding = (lims[1] - lims[0]) * 0.05
    plot_lims = [lims[0] - padding, lims[1] + padding]
    plt.xlim(plot_lims)
    plt.ylim(plot_lims)

    plt.plot(plot_lims, plot_lims, 'k--', alpha=0.75, zorder=0, label='y=x line')


    # Add labels and title
    plt.xlabel(f"{x_col} (KL Divergence)")
    plt.ylabel(f"{y_col} (KL Divergence)")
    plt.title(f"Scatter Plot Comparison: {y_col} vs {x_col}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box') # Make axes equal scale if desired
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'../plots/kl_scatter_comparison_{column_name_1}_{column_name_2}.png')
    # plt.show()
    # plt.close() # Close the figure after showing/saving

def plot_kl_hexbin_comparison(df, x_col='kl_base_ft', y_col='kl_base_ft_intervened'):
    """
    Generates a hexagonal binning plot comparing KL(Base || FT) vs KL(Base || FT Intervened).

    Args:
        df (pd.DataFrame): DataFrame containing KL divergence results with columns
                           'kl_base_ft' and 'kl_base_ft_intervened'.
    """
    plt.figure(figsize=(10, 8)) # Adjusted size slightly for colorbar

    # Create the hexagonal binning plot
    # Adjust gridsize as needed, added cmap and colorbar
    # mincnt=1 hides hexagons with 0 count
    hb = plt.hexbin(df[x_col], df[y_col], gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Point Density') # Add colorbar to show density

    # Determine plot limits based on hexbin boundaries to ensure line covers data
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plot_lims = [min(x_min, y_min), max(x_max, y_max)]

    # Add an identity line (y=x) for reference
    plt.plot(plot_lims, plot_lims, 'r--', alpha=0.75, zorder=1, label='y=x line') # Changed color to red for visibility

    # Set limits back after plotting the line to maintain focus on data
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add labels and title
    plt.xlabel(f"{x_col} (KL Divergence)")
    plt.ylabel(f"{y_col} (KL Divergence)")
    plt.title(f"Hexagonal Binning Comparison: {y_col} vs {x_col}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box') # Keep aspect ratio equal
    plt.tight_layout()
    plt.savefig(f'../plots/kl_hexbin_comparison_{x_col}_{y_col}.png')
    plt.close()
    
def plot_kl_joint_kde_comparison(df, x_col='kl_base_ft', y_col='kl_base_ft_intervened'):
    """
    Generates a joint KDE plot comparing two KL divergence columns,
    with marginal histograms/KDEs.

    Args:
        df (pd.DataFrame): DataFrame containing KL divergence results.
        x_col (str): Name of the column for the x-axis.
        y_col (str): Name of the column for the y-axis.
    """
    # Create the joint plot with KDE in the center and histograms on margins
    # Use fill=True to shade the KDE areas
    g = sns.jointplot(data=df, x=x_col, y=y_col, kind="kde", fill=True, cmap="viridis", height=8)

    # Add the identity line (y=x) to the central KDE plot
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [min(x0, y0), max(x1, y1)]
    g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, zorder=1, label='y=x line')

    # Ensure the axes remain roughly square for visual comparison
    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)
    # If strict equal aspect is needed (might distort marginals):
    # g.ax_joint.set_aspect('equal', adjustable='box')

    # Add a clear overall title
    g.fig.suptitle(f"Joint KDE Plot Comparison: {y_col} vs {x_col}", y=1.02) # Adjust y for title position

    # Add legend to the joint plot axis
    g.ax_joint.legend()

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    plt.savefig(f'../plots/kl_joint_kde_comparison_{x_col}_{y_col}.png')
    plt.close(g.fig) # Close the figure associated with the JointGrid object

def get_top_k_examples(df, k, tokenizer):
    # Sort the DataFrame by kl_act_diff_percent in descending order
    sorted_df = df.sort_values(by='kl_base_ft_intervened', ascending=False)
    
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

    
    # print(top_k_examples, flush=True)
    # Iterate and print each row
    print("Top K Examples:", flush=True)
    top_k_examples.to_csv(f'../csv_files/top_{k}_examples_kl_base_ft_intervened.csv', index=False)
    for index, row in top_k_examples.iterrows():
        print(row.to_string(), flush=True) # Convert row to string for better printing

    # Plot the top k examples
    plt.figure(figsize=(10, 10))
    plt.scatter(top_k_examples['kl_base_ft'], top_k_examples['kl_ft_ft_intervened'], alpha=0.3, s=5, label='Per-Example KL Divergence')
    plt.xlabel("KL(Base || FT)")
    plt.ylabel("KL(Base Reconstructed || FT Reconstructed)")
    plt.title(f"Top {k} Examples by KL Base Reconstructed || FT Reconstructed")
    plt.legend()
    # save the plot
    plt.savefig(f'../plots/top_{k}_kl_comparison.png')
    # plt.show()
    # plt.close()

    return top_k_examples




def get_frequencies(df, tokenizer, column_name_1='kl_base_ft_intervened', column_name_2='kl_base_ft', top_n=100):
    # Sort the DataFrame by kl_act_diff_percent in descending order
    # add new column to df with the difference between the two columns
    
    new_column_name = f'kl_diff_{column_name_1}_{column_name_2}'
    df[new_column_name] = df[column_name_1] - df[column_name_2]
    # filter new_column_name values with values greater than 0.0
    df = df[df[new_column_name] > 0.0]
    # get the frequencies of the appearance of decoded_token in this new df
        # Decode the tokens
    # Ensure the tokenizer is accessible (it's defined globally as base_model.tokenizer)
    try:
        df['decoded_token'] = df['encoded_token'].apply(
            lambda token_id: tokenizer.decode([token_id], skip_special_tokens=False)
        )
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        df['decoded_token'] = df['encoded_token'].astype(str) # Fallback
    df['decoded_token_escaped'] = df['decoded_token'].str.replace('$', r'\$', regex=False)
    frequencies = df['decoded_token_escaped'].value_counts()
    # get the mean of the new_column_name based on decoded_token
    mean_values = df.groupby('decoded_token_escaped')[new_column_name].mean().reset_index()
    # make sure the decoded_token is a string
    mean_values['decoded_token_escaped'] = mean_values['decoded_token_escaped'].astype(str)
    
    # Sort frequencies in descending order
    frequencies = frequencies.sort_values(ascending=False)
    # put frequencies and related decoded_token_escaped values in a csv file
    frequencies.to_csv(f'../csv_files/frequencies_{new_column_name}.csv', index='decoded_token_escaped')

    frequencies = frequencies.sort_values(ascending=False).head(top_n)
    # Align mean_values order with frequencies order (and filter to top N)
    mean_values = mean_values.set_index('decoded_token_escaped').reindex(frequencies.index).reset_index()

    # plot frequencies and mean values in separate subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True) # Increased figure size, share x-axis

    # Plot Frequencies
    axes[0].bar(frequencies.index, frequencies.values, label='Frequency')
    axes[0].set_ylabel('Frequency (Count)')
    axes[0].set_title(f'Top {top_n} Token Frequencies for {new_column_name} > 0') # Updated title
    axes[0].legend()
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot Mean Values
    axes[1].bar(mean_values['decoded_token_escaped'], mean_values[new_column_name], label='Mean Value', color='red', alpha=0.7)
    axes[1].set_xlabel(f'Top {top_n} Decoded Tokens (Sorted by Frequency)') # Updated label
    axes[1].set_ylabel(f'Mean {new_column_name}')
    axes[1].set_title(f'Mean {new_column_name} per Token for Top {top_n} Tokens') # Updated title
    axes[1].legend()
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)
    
    # Adjust layout and save
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(f'../plots/frequencies_and_mean_values_subplots_top_{top_n}_{new_column_name}.png') # Updated filename
    # plt.show()
    plt.close(fig) # Close the figure to free memory



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

def update_example_ids_and_save(input_csv_path, output_csv_path):
    """
    Reads a CSV, halves the 'example_id', and saves to a new CSV.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the modified CSV file.
    """
    try:
        print(f"Reading input CSV from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        print("Modifying 'example_id' column...")
        # Ensure 'example_id' exists and perform integer division by 2
        if 'example_id' in df.columns:
            df['example_id'] = df['example_id'] // 2
        else:
            print("Warning: 'example_id' column not found in the CSV.")
            return

        print(f"Saving updated DataFrame to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False)
        print("Successfully saved the updated CSV.")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_csv = "../csv_files/full_kl_divergence_results_token_wise.csv"
    # output_csv = "../csv_files/full_kl_divergence_results_token_wise_updated_ids.csv"
    # update_example_ids_and_save(input_csv, output_csv)
    #kl_divergence_test(token_wise=True)
    #main()
    df = pd.read_csv(input_csv)
    # Filter the DataFrame
    df = df[df['example_id'] <= 1000]
    # plot_kl_scatter_comparison(df)
    # #plot_kl_act_diff_percent_distribution(df)
    # plot_kl_hexbin_comparison(df) # Add call to the new function
    # # plot_kl_act_diff_percent_distribution(df)
    # # plot_kl_comparison_histograms(df)
    # # k = top 25% of the dataset    
    # k = int(len(df) * 0.10)
    # get_top_k_examples(df, k=k, tokenizer=ft_model.tokenizer) # Example call with k=20
    #get_frequencies(df,tokenizer=ft_model.tokenizer,column_name_1='kl_base_ft_intervened', column_name_2='kl_base_ft', top_n=100)
    #get_frequencies(df, tokenizer=ft_model.tokenizer,column_name_1='kl_ft_ft_intervened', column_name_2='kl_base_ft', top_n=100)
    #plot_kl_hexbin_comparison(df, x_col='kl_base_ft', y_col='kl_base_ft_intervened')
    #plot_kl_hexbin_comparison(df, x_col='kl_base_ft', y_col='kl_ft_ft_intervened')
    plot_kl_joint_kde_comparison(df, x_col='kl_base_ft', y_col='kl_base_ft_intervened')
    plot_kl_joint_kde_comparison(df, x_col='kl_base_ft', y_col='kl_ft_ft_intervened')