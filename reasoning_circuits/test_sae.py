# %%
from sae_lens import SAE
import random
import torch
from nnsight import LanguageModel
import json
from transformers import AutoTokenizer, set_seed
import torch.nn.functional as F
from load_sae import JumpReLUSAE

# %%
GENERATION_SEED = 3  # seed is based on seed used to generate the output text
INIT_SEED = 42

# %%
# Load model and tokenizer
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
print(f"Loading model {MODEL_NAME}...")

model = LanguageModel(
    MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    add_bos_token=True,
)

# %%
# Load SAE with sae_lens

random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

SAE_LAYER = 15
RELEASE = "llama_scope_r1_distill"
SAE_ID = f"l{SAE_LAYER}r_400m_slimpajama_400m_openr1_math"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

sae = SAE.from_pretrained(
    # see other options in sae_lens/pretrained_saes.yaml
    release=RELEASE,
    sae_id=SAE_ID,
    device=DEVICE
)[0]

# %%
# Load the interesting outputs from JSON file
with open("outputs/interesting_outputs.json", "r") as file:
    interesting_outputs = json.load(file)

# input_message = "What's the sum of all proper divisors of 36?"
input_string = interesting_outputs[1]["outputs"][0]["output"]

# %%

# Process the input without applying the chat template
tokenized = tokenizer(
    input_string,
    max_length=1000,
    add_special_tokens=True,  # Add BOS token
    padding=True,
    return_tensors="pt"
)

input_tokens = tokenized['input_ids']

# %%

decoded_input = tokenizer.decode(
    tokenized.input_ids[0], skip_special_tokens=False)
# print(f"Original input: {input}")
print(f"Decoded input: {decoded_input}")

# %%

# Print each token separately with an emoji between them
print("\nTokens with emoji separators:")
for i in range(tokenized.input_ids.shape[1]):
    token_id = tokenized.input_ids[0, i].item()
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"{token_text} ❕", end=" ")

    # Add a newline every 5 tokens for better readability
    if (i + 1) % 5 == 0:
        print()

print("\n")

# %%
# Fetch the activations
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

with torch.no_grad(), model.trace(input_tokens):
    layer = model.model.layers[15]
    original_activation = layer.output[0].clone().save()


# %%
# Encode and decode the activations
random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

encoded_acts = sae.encode(original_activation)
recon_acts = sae.decode(encoded_acts)

# %%
# Check shapes of original and reconstructed activations
print(f"Original activation shape: {original_activation.shape}")
print(f"Reconstructed activation shape: {recon_acts.shape}")

# Check device information
print(f"Original activation device: {original_activation.device}")
print(f"Reconstructed activation device: {recon_acts.device}")

# Check basic statistics
print(f"Original activation - Mean: {original_activation.mean().item():.6f}, Std: {original_activation.std().item():.6f}")
print(f"Reconstructed activation - Mean: {recon_acts.mean().item():.6f}, Std: {recon_acts.std().item():.6f}")

# Check if there are any NaN or Inf values
print(f"Original has NaN: {torch.isnan(original_activation).any().item()}, Inf: {torch.isinf(original_activation).any().item()}")
print(f"Reconstruction has NaN: {torch.isnan(recon_acts).any().item()}, Inf: {torch.isinf(recon_acts).any().item()}")

# %%
# Calculate Mean Squared Error (MSE) loss
mse_loss = F.mse_loss(original_activation,
                      recon_acts.to(original_activation.device))
print(f"Reconstruction MSE Loss: {mse_loss.item()}")

# Optional: Calculate L0 norm (average number of non-zero features per token)
l0_norm = (encoded_acts > 0).float().sum(dim=-1).mean()
print(f"Average L0 norm (sparsity): {l0_norm.item()}")

# %%
# Calculate Zero Reconstruction Baseline MSE
mse_zero_baseline = F.mse_loss(
    original_activation, torch.zeros_like(original_activation))
print(f"Zero Reconstruction Baseline MSE: {mse_zero_baseline.item()}")

# Calculate Mean Reconstruction Baseline MSE
mean_activations = original_activation.mean(
    dim=1, keepdim=True)  # Mean across sequence length
mse_mean_baseline = F.mse_loss(
    original_activation, mean_activations.expand_as(original_activation))
print(f"Mean Reconstruction Baseline MSE: {mse_mean_baseline.item()}")

# %%
# Loss test


def calculate_ce_loss(logits, tokens):
    """Calculates Cross-Entropy loss for language modeling."""
    # Shift logits and labels for next token prediction
    # Logits shape: (batch, seq_len, vocab_size)
    # Tokens shape: (batch, seq_len)
    # Ensure tokens are on the same device as logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens[:, 1:].contiguous().to(
        shift_logits.device)  # Ensure labels are on the same device
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    return loss


random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

# Ensure input_tokens is on the correct device before tracing
input_tokens = input_tokens.to(model.device)

# 1. Original Loss
with torch.no_grad(), model.trace(input_tokens):
    output_logits = model.output.logits.save()  # Logits will be on model.device

original_loss = calculate_ce_loss(
    output_logits, input_tokens)  # Pass device tensors
print(f"Original CE Loss: {original_loss.item()}")

random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

# Ensure recon_acts is on the correct device before intervention
recon_acts = recon_acts.to(model.device)

with torch.no_grad(), model.trace(input_tokens):  # input_tokens is already on device
    layer = model.model.layers[15]
    layer.output[0][:] = recon_acts
    recon_logits = model.output.logits.save()  # Logits will be on model.device

reconstruction_loss = calculate_ce_loss(
    recon_logits, input_tokens)  # Pass device tensors
print(f"Reconstruction CE Loss: {reconstruction_loss.item()}")

random.seed(INIT_SEED)
torch.manual_seed(INIT_SEED)
set_seed(INIT_SEED)

# 3. Zero Ablation Loss
with torch.no_grad(), model.trace(input_tokens):  # input_tokens is already on device
    # Intervene by setting the layer's output to zeros
    # Create zeros based on the *original* activation's shape and dtype
    # Assuming original_activation is already on model.device from a previous forward pass
    zero_acts_intervention = torch.zeros_like(
        original_activation).to(model.device)
    layer = model.model.layers[15]
    layer.output[0][:] = zero_acts_intervention 
    zero_abl_logits = model.output.logits.save()  # Logits will be on model.device

zero_ablation_loss = calculate_ce_loss(
    zero_abl_logits, input_tokens)  # Pass device tensors
print(f"Zero Ablation CE Loss: {zero_ablation_loss.item()}")
# %%
