# %%
import os
import json

# %%
# --- Constants ---
LAYERS = [7, 15, 23]
SAMPLE_ID = 185
CROSSCODER_TYPE = "BatchTopK"  # This will be the subdirectory name

# %%
# --- Functions ---


def compare_lists(list1_name, list1, list2_name, list2):
    set1 = set(list1)
    set2 = set(list2)

    pos_map1 = {val: i + 1 for i, val in enumerate(list1)}
    pos_map2 = {val: i + 1 for i, val in enumerate(list2)}

    intersection_elements = sorted(list(set1.intersection(set2)))
    intersection_with_positions = [
        f"{item} |{pos_map1.get(item, 'N/A')}/{pos_map2.get(item, 'N/A')}|"
        for item in intersection_elements
    ]

    unique_to_list1 = sorted(list(set1.difference(set2)))
    unique_to_list2 = sorted(list(set2.difference(set1)))

    unique1_in_top10_of_list1 = {
        e: i+1 for i, e in enumerate(list1[:10]) if e in unique_to_list1}
    unique2_in_top10_of_list2 = {
        e: i+1 for i, e in enumerate(list2[:10]) if e in unique_to_list2}

    print(f"--- Comparison: {list1_name} vs {list2_name} ---")
    print(
        f"Intersection |sample_rank/mean_rank| ({len(intersection_with_positions)}): {intersection_with_positions}")
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
    """Loads top and bottom item lists from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None, None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Expects lists of dictionaries, where each dict has 'index' and 'value'.
        top_items = data.get('top', [])
        bottom_items = data.get('bottom', [])
        return top_items, bottom_items
    except Exception as e:
        print(f"Error loading or parsing {filepath}: {e}")
        return None, None


def _sort_items_and_extract_indices(items_list):
    """Sorts a list of items by abs(value) and returns their indices."""
    if items_list is None:
        return None
    # Sort by absolute value of 'value', descending.
    # Use .get('value', 0.0) for robustness if 'value' key might be missing.
    try:
        # Ensure all items are dicts for sorting, skip if not (though data should be clean)
        items_list.sort(key=lambda x: abs(x.get('value', 0.0))
                        if isinstance(x, dict) else 0.0, reverse=True)
        # Extract 'index', ensuring item is a dict and has 'index'.
        return [item.get('index') for item in items_list if isinstance(item, dict) and item.get('index') is not None]
    except TypeError:  # Handles cases where items_list might not be sortable as expected
        print(
            f"Warning: Could not sort items_list, it may contain non-dictionary elements or missing 'value'. List: {items_list[:5]}...")
        return None


def main():

    print(
        f"Starting analysis for layers {LAYERS}, Sample ID {SAMPLE_ID}, Type {CROSSCODER_TYPE}\n")

    for layer in LAYERS:
        print(f"--- Processing Layer {layer} ---")

        sample_filename = f"crosscoder_attributions_l{layer}_sample{SAMPLE_ID}_last_token_data.json"
        mean_filename = f"crosscoder_attributions_l{layer}_mean_last_token_data.json"

        # Script is in parent dir, data is in CROSSCODER_TYPE subdir
        sample_filepath = os.path.join(
            f"{CROSSCODER_TYPE}-Crosscoder", sample_filename)
        mean_filepath = os.path.join(
            f"{CROSSCODER_TYPE}-Crosscoder", mean_filename)

        sample_items_top, sample_items_bottom = load_json_data(sample_filepath)
        mean_items_top, mean_items_bottom = load_json_data(mean_filepath)

        sample_top = _sort_items_and_extract_indices(sample_items_top)
        mean_top = _sort_items_and_extract_indices(mean_items_top)
        sample_bottom = _sort_items_and_extract_indices(sample_items_bottom)
        mean_bottom = _sort_items_and_extract_indices(mean_items_bottom)

        if sample_top is not None and mean_top is not None:
            compare_lists(
                f"l{layer}_sample{SAMPLE_ID}_top", sample_top,
                f"l{layer}_mean_top", mean_top
            )
        else:
            print(
                f"Skipping TOP comparison for L{layer} due to missing or unprocessable data.\n")

        if sample_bottom is not None and mean_bottom is not None:
            compare_lists(
                f"l{layer}_sample{SAMPLE_ID}_bottom", sample_bottom,
                f"l{layer}_mean_bottom", mean_bottom
            )
        else:
            print(
                f"Skipping BOTTOM comparison for L{layer} due to missing or unprocessable data.\n")


# %%
if __name__ == "__main__":
    main()

# %%
