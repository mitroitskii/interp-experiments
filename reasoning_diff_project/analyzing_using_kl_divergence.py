from dictionary_learning.dictionary_learning.dictionary import BatchTopKCrossCoder
from torch.nn.functional import cosine_similarity
import torch as th
import numpy as np
import nnsight
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from tqdm import tqdm
import warnings
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
th.set_grad_enabled(False)

import sys
sys.path.append("..")


def calculate_kl_divergence(logits_p, logits_q, temperature=1.0, constant=0):
    """
    Calculate KL divergence between two distributions represented by logits
    
    Args:
        logits_p: Output logits from the first model (shape: batch_size x vocab_size)
        logits_q: Output logits from the second model (shape: batch_size x vocab_size)
        temperature: Temperature for softening the distributions
        
    Returns:
        KL divergence (scalar)
    """
    # Apply temperature scaling
    p_logits = logits_p / temperature
    q_logits = logits_q / temperature
    
    # Convert logits to probabilities
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    # Calculate KL divergence: p(x) * log(p(x)/q(x))
    kl_div = F.kl_div(
        input=th.log(q_probs + constant),
        target=p_probs,
        reduction='batchmean',
        log_target=False
    )
    
    return kl_div

crosscoder_path = "/share/u/models/crosscoder_checkpoints/DeepScaleR_vs_Qwen2.5-Math_L15/ae.pt"
crosscoder = "L15"
extra_args = []
exp_name = "eval_crosscoder"
exp_id = ""
base_layer = 0
reasoning_layer = 1

model_device = "cuda:5"
coder_device = "cuda:6"
calc_device = "cpu"
coder = BatchTopKCrossCoder.from_pretrained(crosscoder_path)
coder = coder.to(coder_device)
num_layers, activation_dim, dict_size = coder.encoder.weight.shape

base_model = nnsight.LanguageModel("Qwen/Qwen2.5-Math-1.5B", device_map=model_device)
ft_model = nnsight.LanguageModel("agentica-org/DeepScaleR-1.5B-Preview",device_map=model_device)
csv_path = Path("csv_files/kl_divergence_results_with_percents.csv")


layer = 15
dataset = load_dataset("koyena/OpenR1-Math-220k-formatted")['train']
dataset = dataset.take(1000)

def main():
    counter = 0
    # Use a list of dictionaries to store results per example
    results = []
    for row in tqdm(dataset):
        # if counter == 25:
        #     continue
        print(f"EXAMPLE NO.{counter}", flush=True)
        print("================================")
        text = row["message_in_chat_template"]
        # print(text)
        # print("================================")
        tokenized_text = ft_model.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text['input_ids']

        # get activations to give to crosscoder
        # get output to measure kl divergence
        # base model
        with base_model.trace(input_ids) as tracer:
            base_activations = base_model.model.layers[layer].output.save()
            base_output = base_model.lm_head.output.save()

        # get activations to give to crosscoder
        # get output to measure kl divergence
        # ft model
        base_output = base_output.to(calc_device)
        with ft_model.trace(input_ids) as tracer:
            ft_activations = ft_model.model.layers[layer].output.save()
            ft_output = ft_model.lm_head.output.save()

        # stack activations retrieved
        activations = th.stack([base_activations[0].to(coder_device), ft_activations[0].to(coder_device)], dim=1)
        # remove the first dimension of size 1 since coder doesn't like that
        activations = activations.squeeze()
        activations = einops.rearrange(activations, 'l t h -> t l h', l=activations.shape[0], t=activations.shape[1], h=activations.shape[2]).to(coder_device)
        # run crosscoder
        crosscoder_activations = coder(activations.float())
        # split crosscoder activations to respective way.
        base_coder = crosscoder_activations[:,0,:].unsqueeze(0)
        ft_coder =  crosscoder_activations[:, 1, :].unsqueeze(0)

        with base_model.edit() as base_model_edited:
            # have to edit it this way since the coder gives out [1536] but there are 159 tokens.
            base_model.model.layers[layer].output[0][:] = base_coder

        with base_model_edited.trace(input_ids) as tracer:
            base_intervened_output = base_model_edited.lm_head.output.save()

        with ft_model.edit() as ft_model_edited:
            ft_model.model.layers[layer].output[0][:] = ft_coder

        with ft_model_edited.trace(input_ids) as tracer:
            ft_intervened_output = ft_model_edited.lm_head.output.save()

        del ft_model_edited
        del base_model_edited
        del base_activations, ft_activations, activations, crosscoder_activations, base_coder, ft_coder
        base_output = base_output.to(calc_device)
        ft_output = ft_output.to(calc_device)   
        base_intervened_output = base_intervened_output.to(calc_device)
        ft_intervened_output = ft_intervened_output.to(calc_device)    
        
        # Inside the loop, before KL calculations
        #print(f"FT output stats: min={th.min(ft_output)}, max={th.max(ft_output)}, has_nan={th.isnan(ft_output).any()}, has_inf={th.isinf(ft_output).any()}", flush=True)
        kl_base_base = calculate_kl_divergence(base_output, base_output).item() # Should be near 0
        kl_base_ft = calculate_kl_divergence(base_output, ft_output).item()
        kl_ft_ft = calculate_kl_divergence(ft_output, ft_output).item() # Should be near 0
        kl_base_base_intervened = calculate_kl_divergence(base_output, base_intervened_output).item()
        kl_ft_ft_intervened = calculate_kl_divergence(ft_output, ft_intervened_output).item()
        print(kl_ft_ft, flush=True)
        del base_output, ft_output

        kl_base_ft_intervened = calculate_kl_divergence(base_intervened_output, ft_intervened_output).item()

        del base_intervened_output, ft_intervened_output        
        kl_act_diff = kl_base_ft - kl_base_ft_intervened # Difference between original and intervened KL
        kl_act_diff_percent = (kl_act_diff / kl_base_ft) * 100

        # Store results for the current example
        current_results = {
            "kl_base_ft": kl_base_ft,
            "kl_base_base": kl_base_base,
            "kl_ft_ft": kl_ft_ft,
            "kl_base_ft_intervened": kl_base_ft_intervened,
            "kl_base_base_intervened": kl_base_base_intervened,
            "kl_ft_ft_intervened": kl_ft_ft_intervened,
            "kl_act_diff": kl_act_diff,
            "kl_act_diff_percent": kl_act_diff_percent,
        }
        results.append(current_results)



        # Clean up tensors to free GPU memory
        
        
        # Delete KL divergence tensors explicitly
        del kl_base_ft, kl_base_base, kl_ft_ft, kl_base_ft_intervened
        del kl_base_base_intervened, kl_ft_ft_intervened, kl_act_diff
        del current_results # Not strictly necessary, but good practice

        # print("================================")
        gc.collect()
        th.cuda.empty_cache()
        counter += 1
        # Check if we need to write the batch to CSV (every 10 examples)
        if (counter + 1) % 10 == 0:
            if results: # Only write if there are results in the batch
                print(f"Writing batch ending at example {counter} to CSV...", flush=True)
                df_batch = pd.DataFrame(results)

                # Calculate normalized values directly in the DataFrame
                for col in df_batch.columns:
                    if col.startswith("kl_"):
                        # Use np.exp for vectorized operation, handle potential NaNs
                        df_batch[f"{col}_norm"] = np.exp(-df_batch[col].astype(float))


                # Reorder columns for clarity (optional)
                norm_cols = sorted([col for col in df_batch.columns if col.endswith("_norm")])
                raw_cols = sorted([col for col in df_batch.columns if col.startswith("kl_") and not col.endswith("_norm")])
                df_batch = df_batch[raw_cols + norm_cols]

                # Append to CSV
                write_header = not csv_path.exists()
                df_batch.to_csv(csv_path, mode='a', header=write_header, index=False)

                # Clear the batch list
                results = []
                print(f"Batch written. Cleared results list.", flush=True)
                del df_batch
                gc.collect()
                th.cuda.empty_cache()   
    



if __name__ == "__main__":
    main()