# %%
import os
import json

# %%
# --- Constants ---
LAYERS = [7, 15, 23]
SAMPLE_ID = 16
CROSSCODER_TYPE = "BatchTopK"  # This will be the subdirectory name

# %%
# --- Functions ---
def compare_lists(list1_name, list1, list2_name, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = sorted(list(set1.intersection(set2)))
    unique_to_list1 = sorted(list(set1.difference(set2)))
    unique_to_list2 = sorted(list(set2.difference(set1)))

    unique1_in_top10_of_list1 = {
        e: i+1 for i, e in enumerate(list1[:10]) if e in unique_to_list1}
    unique2_in_top10_of_list2 = {
        e: i+1 for i, e in enumerate(list2[:10]) if e in unique_to_list2}

    print(f"--- Comparison: {list1_name} vs {list2_name} ---")
    print(f"Intersection ({len(intersection)}): {intersection}")
    print(
        f"Unique to {list1_name} ({len(unique_to_list1)}): {unique_to_list1}")
    if unique1_in_top10_of_list1:
        print(
            f"  >> Unique elements also in top 10 of {list1_name} (element: 1-based position): {unique1_in_top10_of_list1}")
    print(
        f"Unique to {list2_name} ({len(unique_to_list2)}): {unique_to_list2}")
    if unique2_in_top10_of_list2:
        print(
            f"  >> Unique elements also in top 10 of {list2_name} (element: 1-based position): {unique2_in_top10_of_list2}")
    print(
        "-" * (len(f"--- Comparison: {list1_name} vs {list2_name} ---")) + "\n")

def load_json_data(filepath):
    """Loads top and bottom latent indices from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None, None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        top_indices = [item['index'] for item in data.get('top', [])]
        bottom_indices = [item['index'] for item in data.get('bottom', [])]
        return top_indices, bottom_indices
    except Exception as e:
        print(f"Error loading or parsing {filepath}: {e}")
        return None, None

def main():

    print(
        f"Starting analysis for layers {LAYERS}, Sample ID {SAMPLE_ID}, Type {CROSSCODER_TYPE}\n")

    for layer in LAYERS:
        print(f"--- Processing Layer {layer} ---")

        sample_filename = f"crosscoder_attributions_l{layer}_sample{SAMPLE_ID}_last_token_data.json"
        mean_filename = f"crosscoder_attributions_l{layer}_mean_last_token_data.json"

        # Script is in parent dir, data is in CROSSCODER_TYPE subdir
        sample_filepath = os.path.join(f"{CROSSCODER_TYPE}-Crosscoder", sample_filename)
        mean_filepath = os.path.join(f"{CROSSCODER_TYPE}-Crosscoder", mean_filename)

        sample_top, sample_bottom = load_json_data(sample_filepath)
        mean_top, mean_bottom = load_json_data(mean_filepath)

        if sample_top is not None and mean_top is not None:
            compare_lists(
                f"l{layer}_sample{SAMPLE_ID}_top", sample_top,
                f"l{layer}_mean_top", mean_top
            )
        else:
            print(
                f"Skipping TOP comparison for L{layer} due to missing data.\n")

        if sample_bottom is not None and mean_bottom is not None:
            compare_lists(
                f"l{layer}_sample{SAMPLE_ID}_bottom", sample_bottom,
                f"l{layer}_mean_bottom", mean_bottom
            )
        else:
            print(
                f"Skipping BOTTOM comparison for L{layer} due to missing data.\n")


# %%
if __name__ == "__main__":
    main()

# %%
