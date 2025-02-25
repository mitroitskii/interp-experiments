# adapted from https://github.com/jkminder/science-of-finetuning/blob/master/scripts/format_lmsys.py

# %%
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from collections import defaultdict
from argparse import ArgumentParser

source_dataset = "mitroitskii/OpenR1-Math-220k-formatted"
target_dataset = "mitroitskii/OpenR1-Math-220k-formatted"

# %%

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dataset", type=str, default=source_dataset)
    parser.add_argument("--source-path", type=str)
    parser.add_argument("--target-dataset", type=str, default=target_dataset)
    parser.add_argument("--target-path", type=str, default="~/.cache/huggingface/datasets")
    # load the dataset from disk
    parser.add_argument("--tokenizer", type=str, default="agentica-org/DeepScaleR-1.5B-Preview")
    parser.add_argument("--name", "-n", required=True)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
# %%
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# %%    
    if args.source_path:
        ds = load_from_disk(args.source_path)
    else:
        ds = load_dataset(args.source_dataset)

# %%
    formated_messages = defaultdict(list)
    for split in ds.keys():
        dsl = ds[split]["message"]
        for i in tqdm(range(0, len(dsl), 100), desc=f"Processing {split}"):
            
            messages = dsl[i : min(i + 100, len(dsl))]
            
            if tokenizer.bos_token is None:
                formated_messages[split].extend(
                    tokenizer.apply_chat_template(messages, tokenize=False)
                    )
            else:
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)
                assert all(t[0] == tokenizer.bos_token_id for t in tokens)
                tokens = [t[1:] for t in tokens]
                formated_messages[split].extend(tokenizer.batch_decode(tokens))

# %%
    # add the formatted messages to the dataset as a new column
    ds_formatted = ds
    for split in ds.keys():
        ds_formatted[split] = ds_formatted[split].add_column(
            f"message_{args.name}", formated_messages[split]
        )
# %%
    # Save the new dataset to disk
    target_dir = os.path.join(args.target_path, args.target_dataset.split("/")[-1])  # remove the HF username from the path
    ds_formatted.save_to_disk(target_dir)

# %%

    # Push to HF
    ds_formatted.push_to_hub(args.target_dataset)
# %%
