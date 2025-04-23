import sys
from typing import Any, Optional, Tuple, Iterator

import torch as th
from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# from nnsight import LanguageModel # Assuming unused for now
# from datasets import load_from_disk # Assuming unused for now


# %% Load models and tokenizers

# Consider parameterizing these model names
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
FT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def load_models() -> Tuple[
    PreTrainedModel, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizer
]:
    """Load base and fine-tuned models and tokenizers."""
    print(f"Loading fine-tuned model: {FT_MODEL_NAME}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL_NAME, device_map="auto"
    )
    ft_tokenizer = AutoTokenizer.from_pretrained(
        FT_MODEL_NAME, padding_side="left"
    )

    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, device_map="auto"
    )
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    return base_model, ft_model, base_tokenizer, ft_tokenizer


# %% Load dataset


def load_dataset_hf(
    dataset_path: str = "Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B",
    split_part: str = "train",
    subset_size: Optional[int] = 3,
) -> IterableDataset:
    """
    Load dataset from Huggingface Hub.

    Args:
        dataset_path: Path or name of the dataset on Huggingface Hub.
        split_part: Dataset split (e.g., 'train', 'test').
        subset_size: Optional number of samples to take from the dataset.

    Returns:
        The loaded dataset (potentially a subset).
    """
    print(f"Loading dataset: {dataset_path} [{split_part}]")
    dataset = load_dataset(
        dataset_path,
        download_mode="force_redownload", # Be cautious with force_redownload
        streaming=True,
        split=split_part,
    )
    if subset_size is not None:
        print(f"Taking subset of size: {subset_size}")
        subset = dataset.take(subset_size)
        return subset
    else:
        return dataset


# %% Evaluate tokenizer performance


def alternate_test_tokenizer(
    eval_tokenizer: PreTrainedTokenizer,
    dataset: IterableDataset,
    print_stat: str = "Tokenizer",
) -> None:
    """
    Analyze tokenization differences for text with/without <think> tags.
    (Primarily for debugging tokenization).

    Args:
        eval_tokenizer: The tokenizer to test.
        dataset: Dataset containing text samples ('generation' field expected).
        print_stat: Identifier string for print outputs.
    """
    device = "cuda:0" if th.cuda.is_available() else "cpu"
    print(f"--- Running Alternate Tokenizer Test 1 ({print_stat}) ---")
    count = 0
    for data in dataset:
        text = data["generation"]
        without_think_tags = text.replace("<think>", "").replace("</think>", "")

        # Tokenize texts
        with_think = eval_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(device)
        without_think = eval_tokenizer(
            without_think_tags, return_tensors="pt", add_special_tokens=False
        ).to(device)

        # Basic length comparison
        len_with = len(with_think["input_ids"][0])
        len_without = len(without_think["input_ids"][0])
        print(f"Sample {count}: Length With Tags: {len_with}, Without Tags: {len_without}")

        # Optional: Deeper token analysis (uncomment if needed)
        # tokens_with = eval_tokenizer.convert_ids_to_tokens(with_think["input_ids"][0], skip_special_tokens=True)
        # tokens_without = eval_tokenizer.convert_ids_to_tokens(without_think["input_ids"][0], skip_special_tokens=True)
        # print(f"  Tokens (With): {'<>'.join(tokens_with)}")
        # print(f"  Tokens (Without): {'<>'.join(tokens_without)}")

        count += 1
    print(f"--- Finished Alternate Tokenizer Test 1 ({print_stat}) ---")


def alternate_test_tokenizer_2(
    eval_tokenizer: PreTrainedTokenizer,
    decode_tok: PreTrainedTokenizer,
    dataset: IterableDataset,
) -> None:
    """
    Analyze tokenization/decoding differences using two tokenizers.
    (Primarily for debugging cross-tokenizer compatibility).

    Args:
        eval_tokenizer: The tokenizer used for initial encoding.
        decode_tok: The tokenizer used for decoding.
        dataset: Dataset containing text samples ('message_qwen2.5_chat_template' field expected).
    """
    device = "cuda:0" if th.cuda.is_available() else "cpu"
    print("--- Running Alternate Tokenizer Test 2 ---")
    count = 0
    for data in dataset:
        # Assuming specific preprocessing for this test based on original code
        text = data["message_qwen2.5_chat_template"].replace("<｜begin of sentence｜>", "").replace("<｜end of sentence｜>", "")
        without_think_tags = text.replace("<think>", "").replace("</think>", "")

        # Tokenize with eval_tokenizer
        with_think_ids = eval_tokenizer(text, return_tensors="pt").to(device)
        without_think_ids = eval_tokenizer(
            without_think_tags, return_tensors="pt"
        ).to(device)

        # Decode with decode_tok
        decoded_with_think = decode_tok.decode(
            with_think_ids["input_ids"][0], skip_special_tokens=True
        )
        decoded_without_think = decode_tok.decode(
            without_think_ids["input_ids"][0], skip_special_tokens=True
        )

        print(f"Sample {count}:")
        print(f"  Decoded (With Tags): {decoded_with_think}")
        print(f"  Decoded (Without Tags): {decoded_without_think}")

        count += 1
    print("--- Finished Alternate Tokenizer Test 2 ---")


def analyze_generation(
    model: PreTrainedModel,
    eval_tokenizer: PreTrainedTokenizer,
    dataset: IterableDataset,
    num_samples: int = 3, # Note: subset_size in load_dataset_hf controls this now
    with_think_tag: bool = True,
) -> None:
    """
    Generate text using the model and tokenizer, analyzing the output.

    Args:
        model: The model to use for generation.
        eval_tokenizer: The tokenizer to use.
        dataset: Dataset containing prompts ('instruction' field expected).
        num_samples: Number of samples to process (informational, actual limit set by dataset).
        with_think_tag: Whether to append a think tag prompt starter.
    """
    device = model.device
    print(f"--- Analyzing Generation (Think Tags: {with_think_tag}) ---")
    counter = 0

    # Default generation config (can be customized)
    generation_config = {
        "temperature": 0.6, # Use 0 for deterministic output
        "do_sample": True, # Typically False if temperature is 0
        "top_p": 0.95, # Not needed if do_sample=False
        #"repetition_penalty": 1.0,
        "max_new_tokens": 16384, # Set a reasonable limit for generation
    }

    for data in tqdm(dataset, desc="Generating Samples"):
        # Prepare input based on dataset structure (assuming 'instruction' field)
        # This part might need adjustment based on the exact format of 'instruction'
        if isinstance(data.get('instruction'), list) and data['instruction']:
             prompt_content = data['instruction'][0]['content']
        elif isinstance(data.get('instruction'), str):
             prompt_content = data['instruction']
        else:
             print(f"Warning: Skipping sample {counter} due to unexpected instruction format: {data.get('instruction')}")
             continue

        # Apply chat template or format manually
        # Using apply_chat_template is generally preferred if applicable
        # Example: chat = [{"role": "user", "content": prompt_content}]
        # chat_text = eval_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Manual formatting based on original code's likely intent
        chat_text = prompt_content
        if with_think_tag:
            # Ensure space or newline before tag if needed
            chat_text += "<think>" # Or " <think>" depending on model training

        # Tokenize the result
        chat_inputs = eval_tokenizer(chat_text, return_tensors="pt").to(device)

        # Generate output
        generated_ids = model.generate(**chat_inputs, **generation_config)

        # Decode only the newly generated tokens
        output_ids = generated_ids[0][chat_inputs.input_ids.shape[1]:]
        output_text = eval_tokenizer.decode(output_ids, skip_special_tokens=True) # Usually skip special tokens for readability

        print(f"Example No. {counter}")
        print(f"Prompt: {chat_text}")
        print("------------------------------------------------------")
        print(f"Generated Output: {output_text}")
        print("======================================================")
        counter += 1
        # Stop if num_samples is just illustrative for tqdm/dataset.take
        # if counter >= num_samples:
        #     break
    print(f"--- Finished Analyzing Generation (Processed {counter} samples) ---")


# %% Main execution block

def main(num_samples_to_load: int = 3):
    """Main function to load resources and run analyses."""
    print("Loading models...")
    base_model, ft_model, base_tokenizer, ft_tokenizer = load_models()

    print("Loading dataset...")
    # Pass num_samples_to_load to the dataset loader
    dataset = load_dataset_hf(subset_size=num_samples_to_load)

    print("Running generation analysis...")

    # --- Analysis with Base Model ---
    print("Analyzing: Base Model with Base Tokenizer")
    analyze_generation(
        base_model, base_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=False
    )
    # Reset dataset iterator if needed (streaming datasets might be consumed)
    print("Analyzing: Base Model with Fine-Tuned Tokenizer")
    dataset = load_dataset_hf(subset_size=num_samples_to_load)
    analyze_generation(
        base_model, ft_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=True
    )

    # --- Analysis with Fine-Tuned Model ---
    dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    print("Analyzing: Fine-Tuned Model with Base Tokenizer")
    analyze_generation(
        ft_model, base_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=True
    )
    print("Analyzing: Fine-Tuned Model with Fine-Tuned Tokenizer")
    dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    analyze_generation(
        ft_model, ft_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=False
    )
    

    # --- Optional: Cross-Tokenizer Analysis (Uncomment if needed) ---
    # print("Analyzing: Base Model with Fine-Tuned Tokenizer")
    # dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    # analyze_generation(
    #     base_model, ft_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=False
    # )
    # dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    # analyze_generation(
    #     base_model, ft_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=True
    # )

    # print("Analyzing: Fine-Tuned Model with Base Tokenizer")
    # dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    # analyze_generation(
    #     ft_model, base_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=False
    # )
    # dataset = load_dataset_hf(subset_size=num_samples_to_load) # Reset iterator
    # analyze_generation(
    #     ft_model, base_tokenizer, dataset, num_samples=num_samples_to_load, with_think_tag=True
    # )

    # --- Optional: Run Alternate Tokenizer Tests (Uncomment if needed) ---
    # print("Running alternate tokenizer tests (debugging)...")
    # # Need a dataset suitable for alternate_test_tokenizer (with 'generation' field)
    # # debug_dataset_1 = load_dataset_hf(subset_size=num_samples_to_load, ...) # Adjust dataset source if needed
    # # alternate_test_tokenizer(base_tokenizer, debug_dataset_1, print_stat="Base Tokenizer")
    # # alternate_test_tokenizer(ft_tokenizer, debug_dataset_1, print_stat="Fine-Tuned Tokenizer")

    # # Need a dataset suitable for alternate_test_tokenizer_2 (with 'message_qwen2.5_chat_template' field)
    # # debug_dataset_2 = load_dataset_hf(subset_size=num_samples_to_load, ...) # Adjust dataset source if needed
    # # alternate_test_tokenizer_2(base_tokenizer, ft_tokenizer, debug_dataset_2)
    # # alternate_test_tokenizer_2(ft_tokenizer, base_tokenizer, debug_dataset_2)


if __name__ == "__main__":
    # You can add argument parsing here if needed
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_samples", type=int, default=3, help="Number of dataset samples to process")
    # args = parser.parse_args()
    # main(num_samples_to_load=args.num_samples)
    
    main(num_samples_to_load=3) # Default to 3 samples as before