# %% Imports
import argparse
import torch as th
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, load_dataset
import matplotlib.pyplot as plt
import numpy as np
import einops
from tqdm import tqdm
# %% Define RunningMeanStd class for tracking statistics

# from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py


class RunningMeanStd:
    def __init__(self):
        self.mean = None
        self.var = None
        self.count = 0

    def update(self, arr: th.Tensor) -> None:
        batch_mean = arr.double().mean(dim=0)
        batch_var = arr.double().var(dim=0)
        batch_count = arr.shape[0]

        if batch_count == 0:
            return

        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
        else:
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + th.square(delta) * self.count * batch_count /
            (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# %% Load models and tokenizers

def load_models(base_model_name, ft_model_name):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cuda:1",
        torch_dtype=th.bfloat16
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_name,
        device_map="cuda:0",
        torch_dtype=th.bfloat16
    )
    ft_tokenizer = AutoTokenizer.from_pretrained(
        ft_model_name)

    return base_model, ft_model, base_tokenizer, ft_tokenizer

# %% Load dataset


# def load_dataset(dataset_path):
#     return load_from_disk(dataset_path)['test']

# %% Calculate layer-wise differences


def calculate_layer_differences(base_model, ft_model, eval_tokenizer, dataset, num_samples=100):
    device = "cuda" if th.cuda.is_available() else "cpu"
    base_model = base_model.to(device)
    ft_model = ft_model.to(device)

    num_layers = ft_model.config.num_hidden_layers

    # Initialize statistics trackers
    l2_stats = [RunningMeanStd() for _ in range(num_layers)]
    cosine_stats = [RunningMeanStd() for _ in range(num_layers)]

    for idx in tqdm(range(min(num_samples, len(dataset)))):
        input_text = eval_tokenizer.decode(151646) + dataset[idx]['message_in_chat_template']
        inputs = eval_tokenizer(input_text, return_tensors="pt").to(device)

        # Get hidden states from both models
        with th.no_grad():
            base_outputs = base_model(**inputs, output_hidden_states=True)
            ft_outputs = ft_model(**inputs, output_hidden_states=True)
            base_hidden_states = base_outputs.hidden_states
            ft_hidden_states = ft_outputs.hidden_states
            # Calculate differences for each layer
            for layer in range(num_layers):
                # Skip embedding layer
                base_layer = base_hidden_states[layer + 1]
                ft_layer = ft_hidden_states[layer + 1]

                # Flatten
                base_layer_flat = einops.rearrange(
                    base_layer, 'batch seq hidden_dim -> (batch seq) hidden_dim')
                ft_layer_flat = einops.rearrange(
                    ft_layer, 'batch seq hidden_dim -> (batch seq) hidden_dim')

                # L2 norm difference
                l2_diff_per_token = th.norm(
                    base_layer_flat - ft_layer_flat, dim=1)
                l2_stats[layer].update(l2_diff_per_token)

                # Cosine similarity
                cosine_sim_per_token = F.cosine_similarity(
                    base_layer_flat, ft_layer_flat, dim=1)
                cosine_stats[layer].update(cosine_sim_per_token)
    
    return l2_stats, cosine_stats

# %% Plotting functions


def plot_layer_statistics(l2_stats, cosine_stats, base_model_name, ft_model_name):
    num_layers = len(l2_stats)
    layers = range(num_layers)

    # Extract statistics
    l2_means = [stats.mean.mean().item() for stats in l2_stats]
    l2_stds = [th.sqrt(stats.var).mean().item() for stats in l2_stats]
    # Calculate SEM (standard error of the mean) for each layer
    l2_sems = [std / np.sqrt(stats.count)
               for std, stats in zip(l2_stds, l2_stats)]

    # Print some debug info
    print(f"First layer stats:")
    print(f"Mean: {l2_means[0]:.6f}")
    print(f"SEM: {l2_sems[0]:.6f}")

    # 99% confidence intervals (2.576 is the z-score for 99% confidence)
    l2_ci_lower = [mean - 2.576 * sem for mean, sem in zip(l2_means, l2_sems)]
    l2_ci_upper = [mean + 2.576 * sem for mean, sem in zip(l2_means, l2_sems)]

    # Plot L2 norm statistics with confidence intervals
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(layers, l2_means, 'b-', label='Mean')
    plt.fill_between(layers,
                     l2_ci_lower,
                     l2_ci_upper,
                     color='green',    # Change from default color
                     alpha=0.3,       # Increase opacity
                     label='99% CI')

    base_model_name = base_model_name.split('/')[-1]
    ft_model_name = ft_model_name.split('/')[-1]
    plt.title(
        f'L2 Norm Difference by Layer\n(n={l2_stats[0].count} tokens)\n{base_model_name} VS {ft_model_name}')
    plt.xlabel('Layer')
    plt.ylabel('L2 Norm')
    plt.legend()

    cosine_means = [stats.mean.mean().item() for stats in cosine_stats]
    cosine_stds = [th.sqrt(stats.var).mean().item() for stats in cosine_stats]
    # Calculate SEM (standard error of the mean) for each layer
    cosine_sems = [std / np.sqrt(stats.count)
                   for std, stats in zip(cosine_stds, cosine_stats)]

    # 99% confidence intervals (2.576 is the z-score for 99% confidence)
    cosine_ci_lower = [mean - 2.576 * sem for mean,
                       sem in zip(cosine_means, cosine_sems)]
    cosine_ci_upper = [mean + 2.576 * sem for mean,
                       sem in zip(cosine_means, cosine_sems)]

    # Plot Cosine similarity statistics
    plt.subplot(1, 2, 2)
    plt.plot(layers, cosine_means, 'r-', label='Mean')
    plt.fill_between(layers,
                     cosine_ci_lower,
                     cosine_ci_upper,
                     color='green',    # Change from default color
                     alpha=0.3,       # Increase opacity
                     label='99% CI')
    plt.title(
        f'Cosine Similarity by Layer\n(n={cosine_stats[0].count} tokens)\n{base_model_name} VS {ft_model_name}')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f'{base_model_name}_vs_{ft_model_name}-layer-wise-activation-diff.png')
    plt.show()
# %% Main execution


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Compare layer-wise activationdifferences between two models')
    parser.add_argument('--base-model', type=str,
                        default="Qwen/Qwen2.5-Math-1.5B",
                        help='Base model to compare against')
    parser.add_argument('--ft-model', type=str,
                        default="agentica-org/DeepScaleR-1.5B-Preview",
                        help='Fine-tuned model')
    parser.add_argument('--dataset', type=str,
                        default='koyena/OpenR1-Math-220k-formatted',
                        help='Path to the dataset')
    parser.add_argument('--num-samples', type=int,
                        default=1000,
                        help='Number of samples to process')
    args, _ = parser.parse_known_args()

    print("Loading models...")
    base_model, ft_model, _, ft_tokenizer = load_models(
        args.base_model, args.ft_model)

    base_model.eval()
    ft_model.eval()

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)['test']

    print("Calculating layer differences...")
    l2_stats, cosine_stats = calculate_layer_differences(
        base_model, ft_model, ft_tokenizer, dataset, num_samples=args.num_samples
    )

    print("Plotting results...")
    plot_layer_statistics(l2_stats, cosine_stats,
                          args.base_model, args.ft_model)

    # Find most divergent layer
    l2_means = [stats.mean.mean().item() for stats in l2_stats]
    most_divergent_layer = np.argmax(l2_means)
    print(f"\nMost divergent layer (based on L2 norm): {most_divergent_layer}")

if __name__ == "__main__":
    main()
    
# %%
