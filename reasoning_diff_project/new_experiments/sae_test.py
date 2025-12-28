# %%

import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import einops
import nnsight as ns
from nnsight import LanguageModel
import matplotlib.pyplot as plt

BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SAE_REPO_ID = "fnlp/Llama-Scope-R1-Distill"
SAE_SUBFOLDER = "400M-Slimpajama-400M-OpenR1-Math-220k/L15R"
SAE_LAYER = 15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae, device=None):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae, device=device))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model, device=device))
        self.threshold = nn.Parameter(torch.zeros(d_sae, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device))
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device))

        self.d_model = d_model
        self.d_sae = d_sae
        self.device = device

    def encode(self, input_acts):
        input_acts = input_acts.to(self.device)
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold

        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        acts = acts.to(self.device)
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts, return_features: bool = False):
        acts = self.encode(acts)
        recon = self.decode(acts)

        if return_features:
            return recon, acts

        return recon

    @classmethod
    def from_pretrained(cls, repo_id, file_name, device=None):
        """Loads SAE weights from a NumPy (.npy) file downloaded from HF Hub."""
        print(f"Downloading {file_name} from {repo_id}...")
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
        )
        print(f"Downloaded to: {path_to_params}")

        print("Loading numpy parameters...")
        # Assuming it's a saved dictionary
        params = np.load(path_to_params, allow_pickle=True).item()
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        print("Parameters loaded and converted to tensors.")

        # Infer d_model and d_sae from the loaded tensors
        # Assuming standard keys like in the original GemmaScope example
        d_model = pt_params["b_dec"].shape[0]
        d_sae = pt_params["b_enc"].shape[0]
        print(f"Inferred d_model: {d_model}, d_sae: {d_sae}")

        model = cls(d_model, d_sae, device=device)
        # Assumes keys match model parameters directly
        model.load_state_dict(pt_params)
        model.to(device)  # Ensure model is on the correct device
        print("SAE model loaded successfully from NumPy file.")
        return model

    @classmethod
    def from_hf_safetensors(cls, repo_id, subfolder, device=None):
        """
        Loads SAE weights from a Hugging Face Hub safetensors file.
        """
        # Determine the filename within the subfolder
        filename = "sae_weights.safetensors"
        filepath = f"{subfolder}/{filename}"

        print(f"Downloading {filename} from {repo_id}/{subfolder}...")
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=filepath,  # Use the combined path
        )
        print(f"Downloaded to: {path_to_params}")

        print("Loading tensors...")
        # Load directly to target device if possible, convert device to string format
        params = load_file(path_to_params, device=str(
            device) if device else "cpu")
        print("Tensors loaded.")

        # Infer d_model and d_sae from the loaded tensors
        # decoder.bias has shape [d_model]
        # encoder.bias has shape [d_sae]
        d_model = params["decoder.bias"].shape[0]
        d_sae = params["encoder.bias"].shape[0]
        print(f"Inferred d_model: {d_model}, d_sae: {d_sae}")

        # Instantiate the model on the target device
        model = cls(d_model, d_sae, device=device)

        # Create the state dictionary for loading, ensuring tensors are on the model's device
        state_dict = {
            # Transpose encoder weight: safetensors shape (d_sae, d_model) -> desired (d_model, d_sae)
            "W_enc": params["encoder.weight"].T.to(model.device),
            "b_enc": params["encoder.bias"].to(model.device),
            # Transpose decoder weight: safetensors shape (d_model, d_sae) -> desired (d_sae, d_model)
            "W_dec": params["decoder.weight"].T.to(model.device),
            "b_dec": params["decoder.bias"].to(model.device),
            # Calculate threshold from log_jumprelu_threshold
            "threshold": torch.exp(params["log_jumprelu_threshold"]).to(model.device)
        }

        # Load the state dictionary
        model.load_state_dict(state_dict)
        print("SAE model loaded successfully from safetensors.")

        # This final .to(device) is slightly redundant if load_file and state_dict creation worked correctly,
        # but serves as a safeguard.
        model.to(device)

        return model


# %%
sae = JumpReLUSAE.from_hf_safetensors(
    SAE_REPO_ID, SAE_SUBFOLDER, device=DEVICE)

sae.to('cuda')
model = LanguageModel(BASE_MODEL_ID)

text = """"If eating peanuts causes allergic reactions in some people, and Lisa had an allergic reaction, can you conclude she ate peanuts?
Okay, so I'm trying to figure out if I can conclude that Lisa ate peanuts just because she had an allergic reaction. Hmm, let's break this down. First, I know that some people have allergies, right? Like, if someone is allergic to peanuts, eating them can cause reactions like itching, sneezing, or maybe even swelling. So, if Lisa had an allergic reaction, does that automatically mean she ate peanuts?"""
batch_encoding = model.tokenizer(text, return_tensors="pt").to("cuda")
# %%


with model.trace(batch_encoding):
    # get gradients of activations

    layer = model.model.layers[15]
    x = layer.output[0]
    g = x.grad

    sae_latents = ns.apply(sae.encode, x)


    effect = (
        einops.einsum(
            sae.W_dec,
            g,
            "d_sae d_model, batch seq d_model -> batch seq d_sae",
        )
        * sae_latents
    )

    # One line change from other model, just compute effect over the assistant response
    effect = effect.sum(dim=(0, 1)).save()
    logit = model.output.logits
    logit.sum().backward()

# %% Plotting the effect per token
# Remove the debug print for the summed shape
# print(effect.shape, flush=True) 

# Retrieve the saved tensor (per token per feature)
# Assuming batch size is 1, squeeze the batch dimension
#effect_tensor = effect_per_token_feature.value[0] # Shape: (seq, d_sae)

# Sum the effect across all SAE features for each token
effect_per_token = effect.cpu().numpy() # Shape: (seq,)

# Get the tokens corresponding to the input
tokens = model.tokenizer.convert_ids_to_tokens(batch_encoding["input_ids"][0])

# Ensure the number of tokens matches the effect length
if len(tokens) != len(effect_per_token):
    print(f"Warning: Token length ({len(tokens)}) does not match effect length ({len(effect_per_token)}). Truncating/padding effect to match tokens.")
    # Adjust effect_per_token length if necessary (e.g., due to BOS/EOS differences)
    min_len = min(len(tokens), len(effect_per_token))
    tokens = tokens[:min_len]
    effect_per_token = effect_per_token[:min_len]


# Create the plot
plt.figure(figsize=(15, 5))
plt.bar(range(len(tokens)), effect_per_token)
plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
plt.xlabel("Token")
plt.ylabel("Summed Effect across SAE Features")
plt.title(f"Gradient Effect Summed Across SAE Features per Token (Layer 15)") # Add layer info
plt.tight_layout()
plt.show()
plt.savefig(f"../plots/sae/effect_per_token_plot_layer_15.png")
print(f"Plot saved as ../plots/sae/effect_per_token_plot_layer_15.png") # Update save message
