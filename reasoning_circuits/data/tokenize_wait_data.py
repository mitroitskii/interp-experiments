# %%
import json
import os
from transformers import AutoTokenizer

# %%
# --- Constants ---
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# Updated input path to match the output of the previous script
INPUT_JSON_PATH = "wait_occurrences_from_data.json"
# Updated output path name for clarity
OUTPUT_JSON_PATH = "wait_subsequences_from_data.json"

# %%


def find_wait_token_ids(tokenizer):
    """Finds token IDs for ' wait' and ' Wait'."""
    tokens_to_check = ["wait", "Wait", " wait", " Wait"]
    token_ids = set()
    print("Attempting to encode potential 'wait' tokens:")
    for token_str in tokens_to_check:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        print(f"  - '{token_str}' -> IDs: {ids}")
        if len(ids) == 1:
            token_ids.add(ids[0])
        elif len(ids) > 1:
            print(
                f"  - Warning: Token '{token_str}' split into multiple IDs: {ids}. Using first: {ids[0]}")
            token_ids.add(ids[0])  # Use the first token if it splits

    if not token_ids:
        raise ValueError(
            "Could not find distinct token IDs for ' wait' or ' Wait'. Check tokenizer behavior.")
    print(f"Found 'wait'/'Wait' related token IDs: {token_ids}")
    return token_ids

# %%


def main():
    try:
        print(f"Loading tokenizer: {MODEL_NAME}...")
        # Consider adding trust_remote_code=True if needed for specific models
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        wait_token_ids = find_wait_token_ids(tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer or finding wait tokens: {e}")
        return

    # --- Load Input Data ---
    try:
        # Ensure the input path exists
        if not os.path.exists(INPUT_JSON_PATH):
            print(f"Error: Input JSON file not found at '{INPUT_JSON_PATH}'")
            return
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:  # Added encoding
            data = json.load(f)
        print(f"Loaded {len(data)} entries from: {INPUT_JSON_PATH}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {INPUT_JSON_PATH}: {e}")
        return
    except Exception as e:
        print(f"Error loading JSON file {INPUT_JSON_PATH}: {e}")
        return

    if not isinstance(data, list) or not data:
        print("Input JSON is empty or not a list. Exiting.")
        return

    # --- Sanity Check Section (Updated) ---
    print("\n--- Running Sanity Check on First Entry ---")
    first_entry = data[0]
    # Use 'response' instead of 'output'
    first_response_text = first_entry.get('response')
    first_question_text = first_entry.get('question', 'N/A')  # Get question

    if isinstance(first_response_text, str):
        print(
            f"First entry question (preview): {first_question_text[:100]}...")
        print(
            f"First entry response (preview): {first_response_text[:150]}...")
        # Tokenize the response text
        first_token_ids = tokenizer.encode(
            first_response_text, add_special_tokens=True)
        print(f"Sanity Check - Token IDs (first 10): {first_token_ids[:10]}")

        found_wait_indices_sanity = [i for i, token_id in enumerate(
            first_token_ids) if token_id in wait_token_ids]

        if found_wait_indices_sanity:
            print(
                f"Sanity Check: Found 'wait' token(s) at indices: {found_wait_indices_sanity}")
            for idx in found_wait_indices_sanity:
                # Decode surrounding tokens for context
                start_dec = max(0, idx-2)
                end_dec = min(len(first_token_ids), idx+3)
                context_ids = first_token_ids[start_dec:end_dec]
                context_str = tokenizer.decode(context_ids)
                wait_token_id = first_token_ids[idx]
                wait_str = tokenizer.decode([wait_token_id])
                print(
                    f"  - Index {idx}: Wait token ID {wait_token_id} ('{wait_str}'). Context: '{context_str}'")

        else:
            print("Sanity Check: No 'wait' tokens found in the first entry's response.")
    else:
        print("Sanity Check: First entry has missing or non-string 'response' field.")
    print("--- Sanity Check Complete ---\n")
    # --- End Sanity Check ---

    all_subsequences = []
    print("Processing all entries...")
    processed_count = 0
    skipped_count = 0
    for entry_index, entry in enumerate(data):
        # Use 'response' instead of 'output'
        original_response = entry.get('response')
        original_question = entry.get('question')  # Get question

        # Basic validation for the entry structure
        if not isinstance(original_response, str) or not isinstance(original_question, str):
            print(
                f"Warning: Skipping entry {entry_index} due to missing/invalid 'response' or 'question'.")
            skipped_count += 1
            continue

        # Tokenize the response
        token_ids = tokenizer.encode(
            original_response, add_special_tokens=True)

        # Find indices of wait tokens
        wait_indices = [i for i, token_id in enumerate(
            token_ids) if token_id in wait_token_ids]

        if not wait_indices:  # Skip if no wait tokens found in this response
            continue

        # Generate subsequences ending just before each wait token
        for i in wait_indices:
            if i > 0:  # Only create subsequence if wait is not the very first token
                # Subsequence ends *before* the wait token
                subsequence = token_ids[0:i]
                all_subsequences.append({
                    'original_entry_index': entry_index,
                    # Add 'question' field
                    'question': original_question,
                    # Keep preview for context, now from response
                    'original_response_preview': original_response[:100] + "...",
                    'wait_token_index_in_response': i,  # Clarified field name
                    'subsequence_tokens': subsequence
                })
            elif i == 0:
                print(
                    f"Info: Wait token found at index 0 for entry {entry_index}. No preceding subsequence generated.")

        processed_count += 1
        if processed_count % 100 == 0:  # Progress indicator
            print(f"Processed {processed_count} entries...")

    print(f"\nFinished processing.")
    print(f"  Entries processed: {processed_count}")
    print(f"  Entries skipped (missing fields): {skipped_count}")
    print(f"  Generated {len(all_subsequences)} subsequences in total.")

    # --- Save Results ---
    if not all_subsequences:
        print("No subsequences were generated. Not saving output file.")
        return

    try:
        output_dir = os.path.dirname(OUTPUT_JSON_PATH)
        # Check if OUTPUT_JSON_PATH includes a directory part
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:  # Added encoding
            json.dump(all_subsequences, f, indent=2)
        print(
            f"Saved {len(all_subsequences)} subsequences to: {OUTPUT_JSON_PATH}")
    except IOError as e:
        print(f"Error saving results to {OUTPUT_JSON_PATH}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


# %%
if __name__ == "__main__":
    main()

# %%
