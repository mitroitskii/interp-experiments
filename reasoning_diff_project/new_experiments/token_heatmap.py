# %%
# Import libraries
from IPython.display import clear_output
from nnsight import LanguageModel
from typing import List, Callable
import torch
import numpy as np
from IPython.display import clear_output
import plotly.express as px
import plotly.io as pio
# import huggingface tokenizer and model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
import os
is_colab = False

if is_colab:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "plotly_mimetype+notebook_connected+notebook"

# Load model and tokenizer
# %%
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# Load gpt2
model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="auto", dispatch=True, torch_dtype=torch.bfloat16)

USER_TOKEN = 128011
ASSISTANT_TOKEN = 128012
THINK_TOKEN_start = 128013
THINK_TOKEN_END = 128014

USED_DECODED_TOKEN = model.tokenizer.decode(USER_TOKEN)
ASSISTANT_DECODED_TOKEN = model.tokenizer.decode(ASSISTANT_TOKEN)
THINK_DECODED_TOKEN_START = model.tokenizer.decode(THINK_TOKEN_start)
THINK_DECODED_TOKEN_END = model.tokenizer.decode(THINK_TOKEN_END)
# %%

def get_last_layer_subset_indices(data, tokens_to_visualize=["wait", "Wait", " wait", " Wait"]):
    tokens_to_visualize_encoded = model.tokenizer.batch_encode_plus(tokens_to_visualize, add_special_tokens=False)["input_ids"]
    # flatten the list of lists
    tokens_to_visualize_encoded = [item for sublist in tokens_to_visualize_encoded for item in sublist]
    #print(f"Token to visualize encoded: {tokens_to_visualize_encoded}", flush=True)
    layers = model.model.layers
    # print("length of prompt: ", len(model.tokenizer.encode(prompt)))
    prompt = data
    with model.trace() as tracer:
        layer_of_interest = layers[-1]
        with tracer.invoke(prompt) as invoker:
            # Process layer output through the model's head and layer normalization
            layer_output = model.lm_head(model.model.norm(layer_of_interest.output[0]))
            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1)[:, :, tokens_to_visualize_encoded].squeeze(0).save()
    #print(f"probs shape: {probs.shape}")

    # vocab size:
    vocab_size = model.lm_head.out_features
    # for each of the 4 tokens, we want to get the indices where the probability is greater than 0.0
    indices_of_interest = {}
    indices_of_interest_probabilities = {}
    for token_idx in range(len(tokens_to_visualize_encoded)):
        token_name = model.tokenizer.decode(tokens_to_visualize_encoded[token_idx])
        # better than uniformly random value
        probs_greater_than_0 = probs[:, token_idx] > (1.0/vocab_size)
        #probs_greater_than_0 = probs[:, token_idx] > 0.5
        # print(f"probs_greater_than_0 shape: {probs_greater_than_0}")
        # get the indices of the probs_greater_than_0
        indices = torch.where(probs_greater_than_0)[0].detach().cpu().tolist()
        # also gather the probabilities in these indices
        probabilities = probs[indices, token_idx].detach().cpu().tolist()
        # print the probabilities
        #print(f"probabilities: {probabilities}", flush=True)
        indices_of_interest[token_name] = indices
        indices_of_interest_probabilities[token_name] = probabilities
    return indices_of_interest, indices_of_interest_probabilities

def get_frequency_of_tokens(data, indices_of_interest_probabilities=None):
    token_frequency = {}
    for curr_encoded_token, curr_probability in zip(data, indices_of_interest_probabilities):
        if curr_encoded_token not in token_frequency:
            if indices_of_interest_probabilities is not None:
                token_frequency[curr_encoded_token] = curr_probability
            else:
                token_frequency[curr_encoded_token] = 1
        else:
            if indices_of_interest_probabilities is not None:
                token_frequency[curr_encoded_token] += curr_probability
            else:
                token_frequency[curr_encoded_token] += 1

    # sort the token_frequency by value
    token_frequency = dict(sorted(token_frequency.items(), key=lambda item: item[1], reverse=True))

    # decode the keys of token_frequency dict
    token_frequency_decoded = {}
    for key, value in token_frequency.items():
        token_frequency_decoded[model.tokenizer.decode(key)] = value

    del token_frequency

    # sort the token_frequency_decoded by value
    token_frequency_decoded = dict(sorted(token_frequency_decoded.items(), key=lambda item: item[1], reverse=True))
    
    # print the token_frequency_decoded
    for key, value in token_frequency_decoded.items():
        print(f"{key}: {value}")
    
    # return the token_frequency_decoded
    return token_frequency_decoded


def token_heatmap(data, token_to_visualize=["Wait", "wait", " Wait", " wait"], write_html="token_heatmap.html"):
    # generation_config_settings = data["generation_config"]
    #token_to_visualize_encoded = model.tokenizer.encode(token_to_visualize)
    tokens_to_visualize_encoded = model.tokenizer.batch_encode_plus(tokens_to_visualize, add_special_tokens=False)["input_ids"]
    # flatten the list of lists
    tokens_to_visualize_encoded = [item for sublist in tokens_to_visualize_encoded for item in sublist]
    prompt = ""
    # if prompt is already in the format we want, use it
    if isinstance(data, dict):
        if USED_DECODED_TOKEN in data["output"] and THINK_DECODED_TOKEN_START in data["output"]:
            prompt = data["output"]
        else:
            # if the prompt is not in the format we want, we need to add the user, assistant and think tokens
            prompt = USED_DECODED_TOKEN + data["input"] + ASSISTANT_DECODED_TOKEN + THINK_DECODED_TOKEN_START + data["output"]
    else:
        print(f"data is not a dict: {data}", flush=True)
        prompt = data
    #generation_config = GenerationConfig(**generation_config_settings)
    layers = model.model.layers
    probs_layers = []
    print("length of prompt: ", len(model.tokenizer.encode(prompt)))

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = model.lm_head(model.model.norm(layer.output[0]))

                # Apply softmax to obtain probabilities and save the result
                #probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs = torch.nn.functional.softmax(layer_output, dim=-1)[:, :, tokens_to_visualize_encoded].save()
                # take the cumulative sum of the probabilities in probs at the last dimension
                probs_layers.append(probs)

    probs = torch.cat([probs.value for probs in probs_layers])

    # Get the probabilities of a particular token, in this case the token "Wait"
    # particular_token_probs = probs[:, :, token_to_visualize_encoded[-1]]
    # print("last layer wait_probs: ", particular_token_probs[-1])
    # print("wait_probs shape: ", particular_token_probs.shape)
    #max_probs, tokens = probs.max(dim=-1)
    # Decode token IDs to words for each layer
    # words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
    #     for layer_tokens in tokens]

    # print(words)
    # # Access the 'input_ids' attribute of the invoker object to get the input words
    input_ids = invoker.inputs[0][0]["input_ids"][0]
    input_words = [model.tokenizer.decode(t) for t in input_ids]
    input_ids_list = input_ids.cpu().tolist() # Convert tensor to list for hover data
    print(f"Input words (first 10): {input_words[:10]}") # Check initial order
    print(f"Input IDs (first 10): {input_ids_list[:10]}")

    # Convert to float32 before numpy
    particular_token_probs_numpy = probs.detach().cpu().float().numpy()
    print(f"Shape of probability data: {particular_token_probs_numpy.shape}") # Verify shape (layers, tokens)

    # Ensure data dimensions match label length
    
    # take the cumulative sum of the probabilities in particular_token_probs_numpy at the last dimension
    cumulative_probs = np.sum(particular_token_probs_numpy, axis=-1)
    # print the cumulative_probs
    print(f"cumulative_probs shape: {cumulative_probs.shape}", flush=True)
    num_layers, num_tokens = cumulative_probs.shape
    if len(input_words) != num_tokens:
        print(f"Warning: Mismatch between number of words ({len(input_words)}) and data columns ({num_tokens}). Adjusting labels.")
        input_words = input_words[:num_tokens] + ['?'] * (num_tokens - len(input_words))

    fig = px.imshow(
        cumulative_probs,
        x=list(range(num_tokens)),
        y=list(range(num_layers)),
        color_continuous_scale=px.colors.sequential.BuPu,
        color_continuous_midpoint=0.5,
        labels=dict(x="Input Tokens", y="Layers", color="(Cumulative) Probability of different Wait Tokens"),
        aspect="auto"
    )

    # --- Calculate Dynamic Width ---
    # Estimate pixels needed per rotated label (adjust based on font size and desired spacing)
    pixels_per_token_label = 18  # << Tweak this value (e.g., 15-25)
    # Estimate base width needed for y-axis, color bar, margins etc.
    base_width = 200 # << Tweak this value

    calculated_width = max(800, (num_tokens * pixels_per_token_label) + base_width) # Ensure a minimum width

    # --- Update Layout (Use Calculated Width + Rotated Ticks) ---
    fig.update_layout(
        title=f'Visualizing {token_to_visualize} Token Probabilities',
        xaxis_tickangle=-90,
        # --- Use Calculated Width ---
        width=calculated_width,
        height=850, # Keep height fixed or adjust based on calculated_width if desired
        autosize=False,
        # --- Font sizes ---
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(num_tokens)),
            ticktext=input_words,
            tickfont=dict(
                size=11 # The pixels_per_token_label depends on this size
            )
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(num_layers)),
            ticktext=[str(i) for i in range(num_layers)],
            tickfont=dict(
                size=12
            )
        ),
        xaxis_title="Input Tokens",
        yaxis_title="Layers",
        xaxis_title_font=dict(size=16),
        yaxis_title_font=dict(size=16),
        title_font=dict(size=18),
        # --- Keep margins (adjust if needed) ---
        margin=dict(l=50, r=50, b=180, t=80) # Bottom margin is important for rotated labels
    )

    # --- Prepare custom data for hover info ---
    hover_data = [[f"ID: {input_ids_list[x]}<br>Token: {input_words[x]}" for x in range(num_tokens)] for _ in range(num_layers)]

    # --- Update hover template ---
    fig.update_traces(
        customdata=hover_data,
        hovertemplate="<b>Layer</b>: %{y}<br>" +
                      "<b>Token Index</b>: %{x}<br>" +
                      "<b>%{customdata}</b><br>" + # Display custom data (ID and Token)
                      "<b>Probability</b>: %{z:.4f}<extra></extra>" # Format probability
    )

    print(f"Calculated Plot Width: {calculated_width}") # Output the calculated width

    # --- Update Traces for Text (if used) ---
    # fig.update_traces(text=words, texttemplate="%{text}", textfont_size=10)

    fig.write_html(write_html)
# %%

def read_json_file(file_path):
    dataset = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for item in data:
            curr_item = {}
            curr_item["input"] = item["input_message"]
            outputs = item["outputs"]
            for output in outputs:
                curr_item["generation_config"] = output["generation_config"]
                curr_item["output"] = output["output"]
                dataset.append(curr_item)
    return dataset

def read_general_json_file(file_path):
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

def read_txt_file(file_path):
    dataset = []
    with open(file_path, "r") as f:
        # read the entire file as one string
        text = f.read()
        data_items = text.split("--- Sampling")[1:]
        for data_item in data_items:
            curr_item = {}
            # grab text before <|User|>
            sampling_setting = data_item.split(USED_DECODED_TOKEN)[0]
            print(f"Sampling setting: {sampling_setting}", flush=True)
            # strip the sampling setting of any whitespace, equal signs, commas, and parentheses, newlines, and colons
            sampling_setting = sampling_setting.strip().replace("=", "_").replace(",", "_").replace("(", "_").replace(")", "_").replace("\n", "_").replace(":", "_").replace("-", "")
            # remove any trailing or leading whitespace
            sampling_setting = sampling_setting.strip()
            # add to dataset
            curr_item["gen_setting"] = sampling_setting
            curr_item["output"] = USED_DECODED_TOKEN + data_item.split(USED_DECODED_TOKEN)[1]
            dataset.append(curr_item)
    return dataset




if __name__ == "__main__":
    # read json file that has the input and generation config to generate the prompt
    #dataset_name = "../data/interesting_outputs.json"
    #dataset_name = "../data/responses_deepseek-r1-distill-llama-8b.json"
    dataset_name = "/share/u/koyena/crosscoders/wait_occurrences_from_data_manually_cleaned.json"
    # get all files in the data directory
    #data_dir = "/share/u/models/crosscoder_checkpoints/llama8b_outputs/prompt_1"
    # read all files in the data directory
    #files = os.listdir(data_dir)
    # read all files in the data directory
    #file_paths = [os.path.join(data_dir, file) for file in files]
    #html_names = [file.split(".")[0] for file in files]
    file_paths = [dataset_name]
    html_names = ["interesting_outputs"]
    full_data = {}
    full_data_probabilities = {}
    tokens_to_visualize = ["wait", "Wait", " wait", " Wait"]
    #tokens_to_visualize = ["Wait"," Wait"]
    for file_path, html_name in zip(file_paths, html_names):
        dataset_name = file_path
        dataset = []
        # if dataset_name is a json file, read it as a json file
        if dataset_name.endswith(".json"):
            if "responses" or "wait_occurrences" in dataset_name:
                dataset = read_general_json_file(dataset_name)
            else:
                dataset = read_json_file(dataset_name)
        elif dataset_name.endswith(".txt"):
            dataset = read_txt_file(dataset_name)
        else:
            raise ValueError(f"Dataset name {dataset_name} is not a valid file type")
        for item_idx in range(len(dataset)):
            print(f"Processing example {item_idx}")
            example_idx = item_idx
            write_html_path = ""
            if "responses" in dataset_name:
                indices_of_interest, indices_of_interest_probabilities = get_last_layer_subset_indices(dataset[example_idx]["full_response"], tokens_to_visualize=tokens_to_visualize)
                for token_name, indices in indices_of_interest.items():
                    if token_name not in full_data:
                        full_data[token_name] = []
                        full_data_probabilities[token_name] = []
                    full_data[token_name].append(indices)
                    full_data_probabilities[token_name].append(indices_of_interest_probabilities)
            elif "wait_occurrences" in dataset_name:
                # get the indices of the wait token in the full response
                indices_of_interest, indices_of_interest_probabilities = get_last_layer_subset_indices(dataset[example_idx]["response"], tokens_to_visualize=tokens_to_visualize)
                for token_name, indices in indices_of_interest.items():
                    if token_name not in full_data:
                        full_data[token_name] = []
                        full_data_probabilities[token_name] = []
                    full_data[token_name].append(indices)
                    full_data_probabilities[token_name].append(indices_of_interest_probabilities[token_name])
                # os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/wait_occurrences_data/", exist_ok=True)
                # os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/wait_occurrences_data/example_{example_idx}", exist_ok=True)
                # write_html_path = f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/wait_occurrences_data/example_{example_idx}/token_heatmap_{example_idx}.html"
                # token_heatmap(dataset[example_idx]["response"], token_to_visualize=tokens_to_visualize, write_html=write_html_path)
            else:
                token_heatmap(dataset[example_idx], token_to_visualize="Wait", write_html=write_html_path)
            # if item_idx > 2:
            #     break
            # make directory if it doesn't exist
            if False:
                if dataset_name.endswith(".json"):
                    os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/interesting_outputs_data/", exist_ok=True)
                    os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/interesting_outputs_data/example_{example_idx}", exist_ok=True)
                    write_html_path = f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/interesting_outputs_data/example_{example_idx}/token_heatmap_{example_idx}.html"
                else:
                    os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/", exist_ok=True)
                    os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_1/", exist_ok=True)
                    os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_1/{html_name}", exist_ok=True)
                    write_html_path = f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_1/{html_name}/token_heatmap_{dataset[example_idx]['gen_setting']}.html"
                print(f"Writing token heatmap to {write_html_path}", flush=True)
                token_heatmap(dataset[example_idx], token_to_visualize="Wait", write_html=write_html_path)
    # get the frequency of the tokens that give probability greater than 0.01 of particular wait token in the full_data

    # full_data is a list of lists of lists. 
    # The first list is the example index, the second nested list is interested indices for a set of tokens, the third nested list is the indices of the tokens that give probability greater than 0.01 of the particular wait token
    # we want to get the frequency of the tokens that give probability greater than 0.01 of the particular wait token for each interested wait token
    # get the frequency of the tokens that give probability greater than 0.01 of the particular wait token for each interested wait token
    # print(f"full_data: {full_data}", flush=True)
    del dataset
    all_values = []
    all_values_probabilities = []
    for key, value in full_data.items():
        # flatten the list of lists in value
        flattened_value = [item for sublist in value for item in sublist]
        all_values.append(flattened_value)
        flattened_value_probabilities = [item for sublist in full_data_probabilities[key] for item in sublist]
        all_values_probabilities.append(flattened_value_probabilities)
            #print(f"all_values: {all_values}", flush=True)
        # flatten the list of lists in all_values
    flattened_all_values = [item for sublist in all_values for item in sublist]
    flattened_all_values_probabilities = [item for sublist in all_values_probabilities for item in sublist]
    get_frequency_of_tokens(flattened_all_values, flattened_all_values_probabilities)

# %%
