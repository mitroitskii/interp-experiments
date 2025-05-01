# From https://github.com/cadentj/emergent-misalignment/blob/main/notebooks/medical_sae_attribution.py

# %%

from datasets import load_dataset
import torch as t
from torch.utils.data import DataLoader

# %%

from sparsify import Sae
sae = Sae.load_from_hub(
    "kh4dien/sae-Qwen2.5-7B-Instruct-6x", "layers.20", device="cuda"
).to(t.bfloat16)

# %%

def collate_fn(batch):
    messages = [item["messages"] for item in batch]

    temporary = tok.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
        padding=True,
        return_assistant_tokens_mask=True
    )

    messages_mask = temporary["assistant_masks"].bool()

    formatted = tok.apply_chat_template(
        messages,
        tokenize=False
    )

    batch_encoding = tok(
        formatted,
        return_tensors="pt",
        truncation=False,
        padding=True
    )
    return batch_encoding, messages_mask

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)


# %%

from tqdm import tqdm
import einops
import nnsight as ns
from typing import Literal

num_latents = sae.num_latents


def shift_mask(mask, direction: Literal["left", "right"]):
    assert not any(mask[:, -1]), "Last token should be masked"

    if direction == "right":
        shifted = mask.roll(shifts=1, dims=1).clone()

        for row_idx, row in enumerate(shifted):
            for idx in range(len(row) - 1):
                # If [True, False] -> [False, False]
                # Basically clip the ends of messages
                if row[idx] and not row[idx + 1]:
                    shifted[row_idx, idx] = False

        return shifted
    elif direction == "left":
        shifted = mask.roll(shifts=-1, dims=1).clone()

        for row_idx, row in enumerate(shifted):
            clipped = False
            for idx in range(len(row) - 1):

                if clipped:
                    clipped = False
                    continue

                # If [False, True] -> [False, False]
                # Basically clip the starts of messages
                if not row[idx] and row[idx + 1]:
                    shifted[row_idx, idx + 1] = False  # Change is here: modify idx+1, not idx

                    clipped = True
                    continue
                


        return shifted

def compute_effect(model, batch_encoding, messages_mask):
    loss_fn = t.nn.CrossEntropyLoss()

    input_ids = batch_encoding["input_ids"]

    # Shift left because we want to compute the token to the right
    logits_mask = shift_mask(messages_mask, direction="left")


    # Shift right because we want to be predicted
    labels_mask = shift_mask(messages_mask, direction="right")

    with model.trace(batch_encoding):
        # get gradients of activations

        layer = model.model.layers[20]
        x = layer.output[0]
        g = x.grad

        sae_latents = ns.apply(sae.simple_encode, x)

        sae_latents[~messages_mask] = 0
        g[~messages_mask] = 0

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

        logits = model.output.logits[logits_mask]

        target_tokens = input_ids[labels_mask]

        loss = loss_fn(logits, target_tokens)

        loss.backward()

    return effect


effects = t.zeros(num_latents).to("cuda")


for i, (batch_encoding, messages_mask) in tqdm(enumerate(loader)):
    effects += compute_effect(model, batch_encoding, messages_mask)

    if i > 50:
        break

effects /= len(loader)



# %%
import json

idxs = effects.topk(50).indices.tolist()

cache_filter = {
    "model.layers.20" : idxs
}


with open("medical_sae_indices_ce.json", "w") as f:
    json.dump(cache_filter, f)


# %%

with open("medical_sae_indices_ft.json", "r") as f:
    cache_filter = json.load(f)

# %%

import matplotlib.pyplot as plt

plt.plot(effects.sum(dim=1).tolist())