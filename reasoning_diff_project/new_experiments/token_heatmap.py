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

def token_heatmap(data, token_to_visualize="Wait", write_html="token_heatmap.html"):
    # generation_config_settings = data["generation_config"]
    token_to_visualize_encoded = model.tokenizer.encode(token_to_visualize)
    prompt = ""
    # if prompt is already in the format we want, use it
    if USED_DECODED_TOKEN in data["output"] and THINK_DECODED_TOKEN_START in data["output"]:
        prompt = data["output"]
    else:
        # if the prompt is not in the format we want, we need to add the user, assistant and think tokens
        prompt = USED_DECODED_TOKEN + data["input"] + ASSISTANT_DECODED_TOKEN + THINK_DECODED_TOKEN_START + data["output"]
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
                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    probs = torch.cat([probs.value for probs in probs_layers])

    # Get the probabilities of a particular token, in this case the token "Wait"
    particular_token_probs = probs[:, :, token_to_visualize_encoded[-1]]
    print("last layer wait_probs: ", particular_token_probs[-1])
    print("wait_probs shape: ", particular_token_probs.shape)
    #max_probs, tokens = probs.max(dim=-1)
    # Decode token IDs to words for each layer
    # words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
    #     for layer_tokens in tokens]

    # print(words)
    # # Access the 'input_ids' attribute of the invoker object to get the input words
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
    print(f"Input words (first 10): {input_words[:10]}") # Check initial order

    # Convert to float32 before numpy
    particular_token_probs_numpy = particular_token_probs.detach().cpu().float().numpy()
    print(f"Shape of probability data: {particular_token_probs_numpy.shape}") # Verify shape (layers, tokens)

    # Ensure data dimensions match label length
    num_layers, num_tokens = particular_token_probs_numpy.shape
    if len(input_words) != num_tokens:
        print(f"Warning: Mismatch between number of words ({len(input_words)}) and data columns ({num_tokens}). Adjusting labels.")
        input_words = input_words[:num_tokens] + ['?'] * (num_tokens - len(input_words))

    fig = px.imshow(
        particular_token_probs_numpy,
        x=list(range(num_tokens)),
        y=list(range(num_layers)),
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        color_continuous_midpoint=0.50,
        labels=dict(x="Input Tokens", y="Layers", color="Probability of Wait Token"),
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
    # dataset_name = "../data/interesting_outputs.json"

    # get all files in the data directory
    data_dir = "/share/u/models/crosscoder_checkpoints/llama8b_outputs/prompt_4"
    # read all files in the data directory
    files = os.listdir(data_dir)
    # read all files in the data directory
    file_paths = [os.path.join(data_dir, file) for file in files]
    html_names = [file.split(".")[0] for file in files]
    for file_path, html_name in zip(file_paths, html_names):
        dataset_name = file_path
        dataset = []
        # if dataset_name is a json file, read it as a json file
        if dataset_name.endswith(".json"):
            dataset = read_json_file(dataset_name)
        elif dataset_name.endswith(".txt"):
            dataset = read_txt_file(dataset_name)
        else:
            raise ValueError(f"Dataset name {dataset_name} is not a valid file type")
        for item_idx in range(len(dataset)):
            print(f"Processing example {item_idx}")
            example_idx = item_idx
            # make directory if it doesn't exist
            if dataset_name.endswith(".json"):
                os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/interesting_outputs_data/", exist_ok=True)
                os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/interesting_outputs_data/example_{example_idx}", exist_ok=True)
            else:
                os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/", exist_ok=True)
                os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_4/", exist_ok=True)
                os.makedirs(f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_4/{html_name}", exist_ok=True)
            write_html_path = f"/share/u/koyena/www/reasoning_diff_project/token_heatmaps/llama8b_outputs/prompt_4/{html_name}/token_heatmap_{dataset[example_idx]['gen_setting']}.html"
            print(f"Writing token heatmap to {write_html_path}", flush=True)
            token_heatmap(dataset[example_idx], token_to_visualize="Wait", write_html=write_html_path)
# %%
