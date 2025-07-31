# %% Imports
from transformers import AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# %% Load tokenizer


def load_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

# %% Load dataset


def load_dataset():
    return load_from_disk('~/.cache/huggingface/datasets/OpenR1-Math-220k-formatted')['train']

# %% Analyze token statistics


def analyze_token_stats(tokenizer, dataset, field='message_qwen1.5b'):
    """Compute token statistics for the dataset

    Args:
        tokenizer: The tokenizer to use
        dataset: Dataset containing text samples
        field: The field in the dataset to analyze

    Returns:
        dict: Statistics about token counts
    """
    token_counts = []
    total_tokens = 0

    print(f"Analyzing token counts for {len(dataset)} samples...")
    for sample in tqdm(dataset):
        text = sample[field]
        tokens = tokenizer.encode(text)
        count = len(tokens)
        token_counts.append(count)
        total_tokens += count

    stats = {
        'min': np.min(token_counts),
        'max': np.max(token_counts),
        'mean': np.mean(token_counts),
        'median': np.median(token_counts),
        'std': np.std(token_counts),
        'total_samples': len(token_counts),
        'total_tokens': total_tokens
    }

    return stats, token_counts

# %%


def main():
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading dataset...")
    dataset = load_dataset()

    print("Computing token statistics...")
    stats, token_counts = analyze_token_stats(tokenizer, dataset)

    print("\nToken count statistics for 'message_qwen1.5b' field:")
    print(f"Minimum tokens: {stats['min']}")
    print(f"Maximum tokens: {stats['max']}")
    print(f"Average tokens: {stats['mean']:.2f}")
    print(f"Median tokens: {stats['median']}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Total samples analyzed: {stats['total_samples']}")
    print(f"Total tokens in dataset: {stats['total_tokens']:,}")

    # Calculate histogram bins and counts
    num_bins = 50
    counts, bin_edges = np.histogram(token_counts, bins=num_bins)

    # Print bin counts
    print("\nToken count distribution by bin:")
    print(f"{'Bin Range':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for i, count in enumerate(counts):
        bin_start = int(bin_edges[i])
        bin_end = int(bin_edges[i+1])
        bin_range = f"{bin_start}-{bin_end}"
        percentage = (count / stats['total_samples']) * 100
        print(f"{bin_range:<20} {count:<10} {percentage:.2f}%")


if __name__ == "__main__":
    main()

# %%
