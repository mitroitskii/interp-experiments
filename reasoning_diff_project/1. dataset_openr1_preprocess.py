# %%
import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from collections import defaultdict
from argparse import ArgumentParser

# source_dataset = "open-r1/OpenR1-Math-220k"
# target_dataset = "koyena/OpenR1-Math-220k-formatted"
source_dataset = "nvidia/OpenCodeReasoning"
target_dataset = "koyena/OpenCodeReasoning-formatted"
# %%

# Flatten the dataset by exploding list columns


def explode_row_multiple(row, columns_to_explode):
    # Get the length of each list to verify they match
    lengths = []
    for col in columns_to_explode:
        if row[col] is not None:
            lengths.append(len(row[col]))

    if len(set(lengths)) > 1:
        raise ValueError(
            f"All non-None columns to explode must have the same length. Got lengths: {dict(zip(columns_to_explode, lengths))}")

    # Create new rows
    return [{
        **{k: row[k] for k in row if k not in columns_to_explode},
        **{col: row[col][i] if row[col] is not None else None for col in columns_to_explode}
    } for i in range(lengths[0])]


def flatten_dataset(dataset, columns_to_explode):
    return Dataset.from_list(
        [row for row_list in dataset.map(
            lambda x: {'rows': explode_row_multiple(x, columns_to_explode)},
            remove_columns=dataset.column_names,
        )['rows'] for row in row_list]  # ds["rows"] is a list of lists of dicts per each row of the original dataset, so we need a nested loop
    )


# %%
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dataset", type=str, default=source_dataset)
    parser.add_argument("--dataset-config", type=str, default="split_0") # if the dataset has multiple configs
    parser.add_argument("--target-dataset", type=str, default=target_dataset)
    parser.add_argument("--target-path", type=str, default="~/.cache/huggingface/datasets")
    args, _ = parser.parse_known_args()
# %%

    ds = load_dataset(args.source_dataset, args.dataset_config)["split_0"]
    # there is no actual train and test split in the openr1 math dataset
    # ds = ds["train"]
    # ds = ds["split_0"]

# %%
    # drop correctness_count and messages
    #ds = ds.remove_columns(["correctness_count", "messages", 'uuid'])
    ds = ds.remove_columns(['id'])

# %%
    # ds = flatten_dataset(ds, ['generations', 'is_reasoning_complete',
    #                      'correctness_math_verify', 'correctness_llama', 'finish_reasons'])

# %%

    # ds = ds.filter(lambda x: x["is_reasoning_complete"]) # I assume DeepSeek distilled only from complete reasoning traces
# %%
    #ds = ds.rename_column("generations", "generation")
    
    # Reorder columns
    # column_order = [
    #     "problem", "generation", "is_reasoning_complete", "finish_reasons",
    #     "solution", "answer", "correctness_math_verify", "correctness_llama",
    #     "problem_type", "question_type", "source"
    # ]

    # ds = ds.select_columns(column_order)
# %%

    # def add_prefix(item):
    #     item["problem"] = "Please reason step by step, and put your final answer within \\boxed{}." + \
    #         item["problem"]  # this is prepended to deepseek prompts for math reasoning
    #     return item
    # ds = ds.map(add_prefix)

# %%
    # Format problem and generation into message column with user/assistant roles
    messages = [
        [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": generation}
        ]
        for problem, generation in zip(ds["input"], ds["output"])
    ]
    ds = ds.add_column("message", messages)
# %%
    # split and shuffle
    ds = ds.train_test_split(test_size=0.3, seed=42)

# %%

    # Save the new dataset to disk
    target_dir = os.path.join(args.target_path, args.target_dataset.split("/")[-1])  # remove the HF username from the path
    ds.save_to_disk(target_dir)

# %%

    # Push to HF
    ds.push_to_hub(args.target_dataset)

# %%
