#%%
import json
import os

#%%%
# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_filename = "responses_deepseek-r1-distill-llama-8b.json"
output_filename = "responses_deepseek-r1-distill-llama-8b_-extracted_annotated_thinking.json"

input_filepath = os.path.join(current_dir, input_filename)
output_filepath = os.path.join(current_dir, output_filename)

#%%
extracted_responses = []

try:
    # Load the input JSON file
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract 'annotated_thinking' from each entity
    # Assuming the JSON is a list of dictionaries
    if isinstance(data, list):
        for entity in data:
            if isinstance(entity, dict) and 'annotated_thinking' in entity:
                extracted_responses.append(entity['annotated_thinking'])
            else:
                print(f"Warning: Entity does not have 'annotated_thinking' or is not a dict: {entity}")
    else:
        print(f"Warning: Input JSON is not a list. Found type: {type(data)}")
        # Handle other potential structures if necessary, e.g. if data is a dict itself
        # and contains the list of entities under a specific key.
        # For now, we'll assume the top level is a list of entities.

    # Save the extracted responses to a new JSON file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(extracted_responses, f, indent=4)

    print(f"Successfully extracted 'annotated_thinking' values.")
    print(f"Input file: {input_filepath}")
    print(f"Output file: {output_filepath}")
    print(f"Number of responses extracted: {len(extracted_responses)}")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_filepath}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_filepath}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#%%
