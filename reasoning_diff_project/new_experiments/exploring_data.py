# %%
import json
# %%
def read_json_data(file_path):
    """Reads JSON data from the specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list or dict: The loaded JSON data, or None if the file is not found
                      or cannot be decoded.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    
def filter_data(data_json, k=5):
    # Filtering data rows where "annotated_thinking" has a sub-string "[\"backtracking\"] Wait"
    filtered_data = [item for item in data_json if "[\"backtracking\"] Wait" in item["annotated_thinking"]]
    # get top k items where this substring appears earliest.
    top_k_items = sorted(filtered_data, key=lambda x: x["annotated_thinking"].index("[\"backtracking\"] Wait"))[:k]
    # get the "original_message"["content"] and "thinking_response" from the top k items
    # and in "thinking_response", get up to the "Wait" that precedes the "[\"backtracking\"] Wait from the other value ("annotated_response")"
    data_to_return = []
    for item in top_k_items:
        full_response = item["full_response"]
        # find the "Wait" that precedes the "[\"backtracking\"] Wait from the other value ("annotated_response")"
        wait_index = full_response.index("Wait")
        # get the substring of "thinking_response" up to and right before the "Wait"
        full_response_up_to_wait = full_response[:wait_index]
        # add the "original_message" and "thinking_response_up_to_wait" to the item
        data_to_return.append(full_response_up_to_wait)
    # Sort the data_to_return by the length of the strings from shortest to longest and return this sorted list.
    data_to_return.sort(key=len)
    return data_to_return

# def set_data(data_json, k=5):
#     # After running filter_data, get []"original_message"]["content"] and "thinking_response" up o 
# %%
if __name__ == '__main__':
    # Example usage:
    # Replace this with the actual path to your JSON file
    json_file = '../data/responses_deepseek-r1-distill-llama-8b.json' 
    data = read_json_data(json_file)

    #if data:
        #print(f"Successfully read {len(data)} records from {json_file}")
        # You can add more code here to explore the data
        # print(data[0]) # Example: print the first record 
    
    filter_data=filter_data(data, k=1)
    print(filter_data[0])
# %%
