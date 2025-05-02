# %%
import os
import re
import json

# %%

# --- Constants ---
RESULTS_FILENAME = "wait_occurrences_from_outputs.json"
SEED_FILENAME_PREFIX = "seed_"
SEED_FILENAME_SUFFIX = "_results.txt"
SEED_REGEX_PATTERN = rf'{SEED_FILENAME_PREFIX}(\d+){SEED_FILENAME_SUFFIX}'
CONFIG_HEADER_PATTERN = r'^--- (.*?) ---$'
WAIT_WORD_PATTERN = r'\bwait\b'


def find_wait_in_outputs(search_dir):
    """
    Scans seed result files (seed_*.txt) in the specified directory to find
    occurrences of the word 'wait' and collects detailed information.

    Args:
        search_dir (str): The directory containing the seed result files.

    Returns:
        list: A list of dictionaries, where each dictionary contains details
              about an occurrence of 'wait'. Format:
              [{'seed': seed, 'config': config_name, 'output': section_content}]
    """
    results_list = []
    wait_pattern = re.compile(WAIT_WORD_PATTERN, re.IGNORECASE)
    config_pattern = re.compile(CONFIG_HEADER_PATTERN, re.MULTILINE)
    seed_file_pattern = re.compile(SEED_REGEX_PATTERN)

    if not os.path.isdir(search_dir):
        print(f"Error: Search directory '{search_dir}' not found.")
        return results_list

    print(f"Scanning directory: {search_dir}")

    # Iterate through files within the search directory
    for filename in os.listdir(search_dir):
        # Check if the filename matches the expected seed result format
        if filename.startswith(SEED_FILENAME_PREFIX) and filename.endswith(SEED_FILENAME_SUFFIX):
            match = seed_file_pattern.search(filename)
            if not match:
                # This condition might be less likely now but kept for robustness
                print(
                    f"Warning: Skipping file with unexpected format: {filename} in {search_dir}")
                continue

            seed_number = int(match.group(1))
            filepath = os.path.join(search_dir, filename)

            try:
                # Added encoding='utf-8' for broader compatibility
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                continue

            # Find all configuration sections (e.g., "--- Sampling (temp=1.0) ---")
            config_matches = list(config_pattern.finditer(content))

            # Process each configuration section
            for i, match in enumerate(config_matches):
                config_name = match.group(1).strip()
                start_index = match.end()
                # Determine the end of the section
                end_index = config_matches[i+1].start() if (i +
                                                            1) < len(config_matches) else len(content)

                section_content = content[start_index:end_index].strip()

                # Search for the word 'wait' within the section
                if wait_pattern.search(section_content):
                    results_list.append({
                        # 'prompt': prompt_dir_name, # Removed prompt key
                        'seed': seed_number,
                        'config': config_name,
                        'output': section_content
                    })
        # else: # Optional: uncomment to see which files are skipped
            # print(f"Skipping non-matching file: {filename}")

    return results_list


def save_results_to_json(results_list, save_dir):
    """Saves the results list to a JSON file in the specified directory."""
    if not results_list:
        print("\nNo occurrences of 'wait' found. No JSON file saved.")
        return

    # Sort results for consistent output (removed 'prompt' from key)
    results_list.sort(key=lambda x: (x['seed'], x['config']))

    # Save the JSON file in the specified save directory
    json_file_path = os.path.join(save_dir, RESULTS_FILENAME)
    try:
        # Added encoding='utf-8'
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2)
        print(f"\nJSON results saved to: {json_file_path}")
    except Exception as e:
        print(f"Error saving JSON file {json_file_path}: {e}")


# %%
if __name__ == "__main__":
    # Use the current working directory for searching and saving
    current_working_directory = os.getcwd()
    print(
        f"Searching for '{SEED_FILENAME_PREFIX}*{SEED_FILENAME_SUFFIX}' files in: {current_working_directory}")
    print(
        f"Results will be saved as '{RESULTS_FILENAME}' in: {current_working_directory}")

    # Find occurrences of 'wait'
    wait_occurrences_list = find_wait_in_outputs(current_working_directory)

    # Save the results to JSON
    save_results_to_json(wait_occurrences_list, current_working_directory)

# %%
