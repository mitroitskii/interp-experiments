# %%
import os
import re
import json

# %%

# --- Constants ---
# Get the current working directory (where the script is executed)
CURRENT_WORKING_DIR = os.getcwd()

# Define filenames, assuming they are in the CURRENT_WORKING_DIR
INPUT_JSON_FILENAME = "responses_deepseek-r1-distill-llama-8b.json"  # Just the filename
# Changed filename to reflect content
RESULTS_FILENAME = "wait_occurrences_from_data.json"

# Construct full paths based on the CURRENT_WORKING_DIR
INPUT_JSON_PATH = os.path.join(CURRENT_WORKING_DIR, INPUT_JSON_FILENAME)
OUTPUT_JSON_PATH = os.path.join(CURRENT_WORKING_DIR, RESULTS_FILENAME)

WAIT_WORD_PATTERN = r'\bwait\b'


def find_wait_in_json(input_json_path):
    """
    Scans a JSON file containing a list of objects. For objects where
    the 'full_response' field contains 'wait', it extracts the question
    (from original_message.content) and the response (full_response).

    Args:
        input_json_path (str): The path to the input JSON file.
                               The file should contain a JSON array of objects.

    Returns:
        list: A list of dictionaries, where each dictionary has 'question'
              and 'response' keys. Returns an empty list if the file is
              not found, is not valid JSON, no occurrences are found, or
              required nested keys are missing.
    """
    results_list = []  # Will store {'question': ..., 'response': ...} dicts
    wait_pattern = re.compile(WAIT_WORD_PATTERN, re.IGNORECASE)

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at '{input_json_path}'")
        return results_list
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'")
        return results_list
    except Exception as e:
        print(f"Error reading file {input_json_path}: {e}")
        return results_list

    if not isinstance(data, list):
        print(f"Error: Input JSON does not contain a list (array) at the top level.")
        return results_list

    for i, item in enumerate(data):  # Use enumerate for better warning messages
        if not isinstance(item, dict):
            print(f"Warning: Skipping non-object item at index {i}: {item}")
            continue

        full_response = item.get("full_response")

        if isinstance(full_response, str):
            if wait_pattern.search(full_response):
                # Extract nested question content
                original_message = item.get("original_message")
                question_content = None
                if isinstance(original_message, dict):
                    question_content = original_message.get("content")

                # Check if we successfully extracted the question
                if isinstance(question_content, str):
                    results_list.append({
                        "question": question_content,
                        "response": full_response
                    })
                else:
                    print(
                        f"Warning: Skipping item at index {i} due to missing/invalid 'original_message' or 'content'. Response started with: '{full_response[:50]}...'")

        # else: # Optional: Warn if 'full_response' is missing/not string
            # print(f"Warning: Item at index {i} has missing or non-string 'full_response'.")

    return results_list


def save_results_to_json(results_list, output_json_path):
    """Saves the results list (list of {'question': ..., 'response': ...}) to a JSON file."""
    if not results_list:
        print("\nNo occurrences of 'wait' found in 'full_response' fields with valid questions. No output file saved.")
        return

    output_dir = os.path.dirname(output_json_path)

    if os.path.isdir(output_json_path):
        print(
            f"Error: Output path '{output_json_path}' is a directory. This should not happen.")
        return

    # Optional: Sort results based on question or response if desired
    # try:
    #     results_list.sort(key=lambda x: x.get('question', '')) # Sort by question
    # except Exception as e:
    #     print(f"Warning: Could not sort results: {e}")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)
        print(
            f"\nJSON results (list of question/response pairs) saved to: {output_json_path}")
    except IOError as e:
        print(f"Error saving JSON file to {output_json_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


def main():
    """Uses constants to find wait occurrences, extract question/response pairs, and save results."""
    print(f"Looking for input file: {INPUT_JSON_PATH}")
    wait_occurrences_list = find_wait_in_json(INPUT_JSON_PATH)  # List of dicts

    if wait_occurrences_list is not None:
        print(
            f"Saving extracted question/response pairs to: {OUTPUT_JSON_PATH}")
        save_results_to_json(wait_occurrences_list, OUTPUT_JSON_PATH)

# %%


if __name__ == "__main__":
    main()

# %%
