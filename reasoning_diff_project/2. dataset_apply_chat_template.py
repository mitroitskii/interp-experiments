# adapted from https://github.com/jkminder/science-of-finetuning/blob/master/scripts/format_lmsys.py

# %%
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from collections import defaultdict
from argparse import ArgumentParser

source_dataset = "koyena/OpenCodeReasoning-formatted"
target_dataset = "koyena/OpenCodeReasoning-formatted"

# For QWEN
BOS = 151646
USER = 151644
ASSISTANT = 151645
NEWLINE = 198
THINK_START = 151648
THINK_END = 151649
EOS = 151643

# %%

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source-dataset", type=str, default=source_dataset)
    parser.add_argument("--source-path", type=str)
    parser.add_argument("--target-dataset", type=str, default=target_dataset)
    parser.add_argument("--target-path", type=str, default="~/.cache/huggingface/datasets")
    # load the dataset from disk
    parser.add_argument("--tokenizer", type=str, default="agentica-org/DeepCoder-1.5B-Preview")
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
            manual_formatted_messages = []
            # we are not including BOS token
            for m in messages:
                new_m = tokenizer.decode(USER) + m[0]["content"] + tokenizer.decode(ASSISTANT) + m[-1]["content"] + tokenizer.decode(EOS)
                manual_formatted_messages.append(new_m)

            formated_messages[split].extend(manual_formatted_messages)            
            # if tokenizer.bos_token is None:
            #     formated_messages[split].extend(
            #         tokenizer.apply_chat_template(messages, tokenize=False)
            #         )
            # else:
            #     tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=True)
            #     assert all(t[0] == tokenizer.bos_token_id for t in tokens)
            #     tokens = [t[1:] for t in tokens]
            #     formated_messages[split].extend(tokenizer.batch_decode(tokens))
            
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
