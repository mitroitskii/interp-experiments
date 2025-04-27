# %%

import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import einops

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
model = JumpReLUSAE.from_hf_safetensors(
    SAE_REPO_ID, SAE_SUBFOLDER, device=DEVICE)