# %%
import json
import os
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
INPUT_JSON_PATH = "wait_occurrences_from_outputs.json"
OUTPUT_JSON_PATH = "wait_subsequences_from_outputs.json"

# %%

# NOTE: some of the first results repeat because I had duplicating generation configs for prompt 1

def find_wait_token_ids(tokenizer):
    """Finds token IDs for ' wait' and ' Wait'."""
    tokens_to_check = ["wait", "Wait", " wait", " Wait"]
    token_ids = set()
    print("Attempting to encode potential 'wait' tokens:")  # Debug
    for token_str in tokens_to_check:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        print(f"  - '{token_str}' -> IDs: {ids}")  # Debug
        if len(ids) == 1:
            token_ids.add(ids[0])
        elif len(ids) > 1:
            # This warning might be important if 'wait' isn't a single token sometimes
            print(
                f"  - Warning: Token '{token_str}' split into multiple IDs: {ids}. Adding first: {ids[0]}")
            token_ids.add(ids[0])

    if not token_ids:
        raise ValueError("Could not find token IDs for 'wait' or 'Wait'.")
    print(f"Found 'wait'/'Wait' related token IDs: {token_ids}")
    return token_ids

# %%


def main():
    try:
        print(f"Loading tokenizer: {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        wait_token_ids = find_wait_token_ids(tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer or finding wait tokens: {e}")
        return
    try:
        with open(INPUT_JSON_PATH, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from: {INPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error loading JSON file {INPUT_JSON_PATH}: {e}")
        return

    if not data:
        print("Input JSON is empty. Exiting.")
        return

    # --- Sanity Check Section ---
    print("\n--- Running Sanity Check on First Entry ---")
    first_entry = data[0]
    first_output_text = first_entry.get('output')
    if first_output_text:
        print(
            f"First entry prompt: {first_entry.get('prompt')}, seed: {first_entry.get('seed')}, config: {first_entry.get('config')}")
        print(f"First entry output (preview): {first_output_text[:150]}...")
        first_token_ids = tokenizer.encode(
            first_output_text, add_special_tokens=True)
        # Debug
        print(f"Sanity Check - Token IDs (first 10): {first_token_ids[:10]}")
        found_wait_indices_sanity = [i for i, token_id in enumerate(
            first_token_ids) if token_id in wait_token_ids]

        if found_wait_indices_sanity:
            print(
                f"Sanity Check: Found 'wait' token(s) at indices: {found_wait_indices_sanity}")
            for idx in found_wait_indices_sanity:
                if idx > 0:
                    preceding_token_id = first_token_ids[idx-1]
                    wait_token_id = first_token_ids[idx]
                    print(
                        f"  - Index {idx}: Preceding token ID {preceding_token_id} ('{tokenizer.decode([preceding_token_id])}'), Wait token ID {wait_token_id} ('{tokenizer.decode([wait_token_id])}')")
                else:
                    wait_token_id = first_token_ids[idx]
                    print(
                        f"  - Index {idx}: Wait token ID {wait_token_id} ('{tokenizer.decode([wait_token_id])}') at the beginning.")
        else:
            print("Sanity Check: No 'wait' tokens found in the first entry's output.")
    else:
        print("Sanity Check: First entry has no 'output' field.")
    print("--- Sanity Check Complete ---\n")
    # --- End Sanity Check ---

    all_subsequences = []
    print("Processing all entries...")
    for entry_index, entry in enumerate(data):
        original_text = entry.get('output')
        if not original_text:
            print(
                f"DEBUG: Skipping entry {entry_index} due to missing 'output'.")
            continue

        # Debug: Print info about the entry being processed in the main loop
        if entry_index == 0:
            print(
                f"DEBUG: Processing first entry (index {entry_index}) in main loop...")
            print(f"DEBUG: Text (preview): {original_text[:150]}...")

        token_ids = tokenizer.encode(original_text, add_special_tokens=True)
        if entry_index == 0:
            # Debug
            print(f"DEBUG: Main Loop - Token IDs (first 10): {token_ids[:10]}")

        wait_indices = [i for i, token_id in enumerate(
            token_ids) if token_id in wait_token_ids]

        if entry_index == 0:  # Debug specifically for the first entry
            print(
                f"DEBUG: Main loop found wait_indices for entry 0: {wait_indices}")

        if not wait_indices:  # Skip if no wait tokens found in this entry
            continue

        for i in wait_indices:
            # This is the block that *should* be executing for index 762 in the first entry
            if i > 0:
                subsequence = token_ids[0:i]
                # Critical Debug Point: Check if this line is printed for the first entry
                print(
                    f"DEBUG: Main Loop - Appending subsequence for entry {entry_index}, wait index {i}. Subsequence length: {len(subsequence)}")
                all_subsequences.append({
                    'original_entry_index': entry_index,
                    'prompt': entry.get('prompt'),
                    'seed': entry.get('seed'),
                    'config': entry.get('config'),
                    'original_output_preview': original_text[:100] + "...",
                    'wait_token_index_in_original': i,
                    'subsequence_tokens': subsequence
                })
            elif i == 0:
                print(
                    f"DEBUG: Main Loop - Wait token found at index 0 for entry {entry_index}. No preceding subsequence.")
            else:
                # This case should not happen if i is an index >= 0
                print(
                    f"DEBUG: Main Loop - Unexpected index {i} for entry {entry_index}.")

    print(
        f"Finished processing. Found {len(all_subsequences)} subsequences in total.")

    # Save the results
    try:
        output_dir = os.path.dirname(OUTPUT_JSON_PATH)
        if output_dir:  # Ensure output directory exists if specified
            os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_subsequences, f, indent=2)
        print(
            f"Saved {len(all_subsequences)} subsequences to: {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"Error saving results to {OUTPUT_JSON_PATH}: {e}")


# %%
if __name__ == "__main__":
    main()

# %%
