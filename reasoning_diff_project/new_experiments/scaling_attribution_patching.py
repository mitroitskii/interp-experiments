from IPython.display import clear_output
import nnsight
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected+colab+notebook"
from nnsight import LanguageModel, util
import einops
import torch
import os
llm = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", torch_dtype=torch.bfloat16)
#print(llm)
old_prompt = """If eating peanuts causes allergic reactions in some people, and Lisa had an allergic reaction, can you conclude she ate peanuts?
Okay, so I'm trying to figure out if I can conclude that Lisa ate peanuts just because she had an allergic reaction. Hmm, let's break this down. First, I know that some people have allergies, right? Like, if someone is allergic to peanuts, eating them can cause reactions like itching, sneezing, or maybe even swelling. So, if Lisa had an allergic reaction, does that automatically mean she ate peanuts?"""

prompt = """Create a barter system for a community of people with highly diverse skills and needs.
Okay, so I need to create a barter system for a community with a lot of different skills and needs. Hmm, where do I start? I guess the first thing is to figure out what exactly the community needs. Maybe I should list out all the skills people have and what they can offer. That makes sense. So, I'll need a way to categorize the skills, like maybe divide them into categories like arts, crafts, services, etc. """

torch.manual_seed(42)
# encode the prompt
prompt_encoded =   llm.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
prompt_noisy = prompt_encoded.clone()
# decode the noisy prompts
prompt_noisy_decoded = llm.tokenizer.decode(prompt_noisy)
noisy_answer = "Wait"

def get_logit_diff(logits, answer_token_indices):
    logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

def run_attribution_patching(prompt, prompt_noisy_decoded, answers, example_id=0, at_token_index=0):
    # Tokenize clean and corrupted inputs:
    clean_tokens = llm.tokenizer(prompt, return_tensors="pt")["input_ids"]
    corrupted_tokens = llm.tokenizer(prompt_noisy_decoded, return_tensors="pt")["input_ids"]
    # corrupted_tokens = clean_tokens[
    #     [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    # ]

    # Tokenize answers for the single prompt:
    # Assumes answers[0] is the correct answer, answers[1] is the incorrect answer.
    # Also assumes the actual token ID is the second element ([1]) after tokenization (might need adjustment if tokenizer behaves differently).
    correct_answer_token_id = llm.tokenizer(answers[0])["input_ids"][1]
    incorrect_answer_token_id = llm.tokenizer(answers[1])["input_ids"][1]
    answer_token_indices = torch.tensor([[correct_answer_token_id, incorrect_answer_token_id]]) # Shape will be (1, 2)

    print("answer_tokens = " , answer_token_indices, flush=True)
    clean_logits = llm.trace(clean_tokens, trace=False).logits.cpu()
    corrupted_logits = llm.trace(corrupted_tokens, trace=False).logits.cpu()

    CLEAN_BASELINE = get_logit_diff(clean_logits, answer_token_indices).item()
    print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")

    CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_token_indices).item()
    print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")
    
    def ioi_metric(
            logits,
            answer_token_indices=answer_token_indices,
            ):
         return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
             CLEAN_BASELINE - CORRUPTED_BASELINE
             )
    print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
    print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

    clean_out = []
    corrupted_out = []
    corrupted_grads = []


    mlp_clean_out = []
    mlp_corrupted_out = []
    mlp_corrupted_grads = []

    entire_residual_clean_out = []
    entire_residual_corrupted_out = []
    entire_residual_corrupted_grads = []

    with llm.trace() as tracer:
    # Using nnsight's tracer.invoke context, we can batch the clean and the
    # corrupted runs into the same tracing context, allowing us to access
    # information generated within each of these runs within one forward pass

        with tracer.invoke(clean_tokens) as invoker_clean:
        # need to set requires grad to true for remote
            #llm.model.layers[0].self_attn.o_proj.input.requires_grad = True
            # Gather each layer's attention
            for layer in llm.model.layers:
                # Get clean attention output for this layer
                # across all attention heads
                attn_out = layer.self_attn.o_proj.input
                mlp_out = layer.mlp.down_proj.input
                entire_residual_out = layer.input

                clean_out.append(attn_out.save())
                mlp_clean_out.append(mlp_out.save())
                entire_residual_clean_out.append(entire_residual_out.save())

        with tracer.invoke(corrupted_tokens) as invoker_corrupted:
            # Gather each layer's attention and gradients
            for layer in llm.model.layers:
                # Get corrupted attention output for this layer
                # across all attention heads
                attn_out = layer.self_attn.o_proj.input
                mlp_out = layer.mlp.down_proj.input
                entire_residual_out = layer.input
                corrupted_out.append(attn_out.save())
                mlp_corrupted_out.append(mlp_out.save())
                entire_residual_corrupted_out.append(entire_residual_out.save())

                # save corrupted gradients for attribution patching
                corrupted_grads.append(attn_out.grad.save())
                mlp_corrupted_grads.append(mlp_out.grad.save())
                entire_residual_corrupted_grads.append(entire_residual_out.grad.save())
            # Let's get the logits for the model's output
            # for the corrupted run
            logits = llm.lm_head.output.save()

            # Our IOI metric uses tensors saved on cpu, so we
            # need to move the logits to cpu.
            value = ioi_metric(logits.cpu())

            # We also need to run a backwards pass to
            # update gradient values
            value.backward()

    patching_results = []

    for corrupted_grad, corrupted, clean, layer in zip(
        corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
    ):

        residual_attr = einops.reduce(
            corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
            "batch (head dim) -> head",
            "sum",
            head = 32,
            dim = 128,
        )

        patching_results.append(
            residual_attr.detach().cpu().float().numpy()
        )

    fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Attention Heads",
    labels={"x": "Head", "y": "Layer","color":"Norm. Logit Diff"},
    )

    # make folder if not exists
    
    os.makedirs(f'../attr_patching_plots_cumulative/example_{example_id}', exist_ok=True)
    os.makedirs(f'../attr_patching_plots_cumulative/example_{example_id}/token_{at_token_index}', exist_ok=True)

    plot_attribution_patching(corrupted_grads, corrupted_out, clean_out, example_id, at_token_index, position_type="attention head", component_type="attn")
    plot_attribution_patching(mlp_corrupted_grads, mlp_corrupted_out, mlp_clean_out, example_id, at_token_index, position_type="tokens", component_type="mlp")
    plot_attribution_patching(entire_residual_corrupted_grads, entire_residual_corrupted_out, entire_residual_clean_out, example_id, at_token_index, position_type="tokens", component_type="residual")


def plot_attribution_patching(corrupted_grads, corrupted, clean, example_id, at_token_index, position_type="tokens", component_type="attn", folder_name="attr_patching_plots_cumulative"):
    patching_results = []

    for corrupted_grad, corrupted, clean, layer in zip(
        corrupted_grads, corrupted, clean, range(len(clean))
    ):

        residual_attr = einops.reduce(
            corrupted_grad.value * (clean.value - corrupted.value),
            "batch pos dim -> pos",
            "sum",
        )

        patching_results.append(
            residual_attr.detach().cpu().float().numpy()
        )
    fig = px.imshow(
        patching_results,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        title=f"Attribution Patching Over {position_type} ({component_type})",
        labels={"x": "{position_type}", "y": "Layer","color":"Norm. Logit Diff"},

    )

    fig.update_xaxes(dtick=1) # Show ticks for every token position

    print(f"Writing {component_type} w.r.t. {position_type} plot for token {at_token_index}...")
    fig.write_image(file=f'../{folder_name}/example_{example_id}/token_{at_token_index}/{component_type}_{position_type}.png', format='png')

# for i in range(2, len(prompt_encoded)):
#     prompt_noisy = prompt_encoded.clone()
#     prompt_noisy[i] = prompt_encoded[i] + torch.randint(0, 10000, (1,))
#     # decode the noisy prompts
#     prompt_noisy_decoded = llm.tokenizer.decode(prompt_noisy)
#     with llm.generate(prompt_noisy_decoded, max_new_tokens=20) as tracer:
#         out = llm.generator.output.save()

#     # Adding print statements to see prompt, noisy prompt, and noisy answer
#     print("Prompt: ", prompt)
#     print("Noisy Prompt: ", prompt_noisy_decoded)
#     print("Noisy Answer: ", noisy_answer)
#     print("--------------------------------", flush=True)
    
#     noisy_answer = llm.tokenizer.decode(out[0][len(prompt_noisy):])
#     # if "Wait" not in noisy_answer:
#     #     print("Found the answer at example: ", i)
#     #     run_attribution_patching(prompt, prompt_noisy_decoded, ["Wait", noisy_answer], example_id=0, at_token_index=i)

# New loop: Add noise cumulatively from the last token to the first
print("\nStarting cumulative noise addition loop...")
prompt_noisy_cumulative = prompt_encoded.clone()
for j in range(len(prompt_encoded) - 1, -1, -1):  # Iterate backwards
    # Add noise to the current token (index j) on top of existing noise
    #prompt_noisy_cumulative = prompt_encoded.clone()
    prompt_noisy_cumulative[j] = prompt_noisy_cumulative[j] + torch.randint(0, 10000, (1,))
    
    # decode the noisy prompts
    prompt_noisy_decoded = llm.tokenizer.decode(prompt_noisy_cumulative)
    
    with llm.generate(prompt_noisy_decoded, max_new_tokens=30) as tracer:
        out = llm.generator.output.save()

    noisy_answer = llm.tokenizer.decode(out[0][len(prompt_noisy_cumulative):])
    print(f"Iteration j={j}:")
    print("Prompt: ", prompt)
    print("Cumulative Noisy Prompt: ", prompt_noisy_decoded)
    print("Cumulative Noisy Answer: ", noisy_answer)
    print("--------------------------------", flush=True)

    # Adding print statements to see prompt, noisy prompt, and noisy answer
    # if j == 69:
    #     print(f"Iteration j={j}:")
    #     print("Prompt: ", prompt)
    #     print("Cumulative Noisy Prompt: ", prompt_noisy_decoded)
    #     print("Cumulative Noisy Answer: ", noisy_answer)
    #     print("--------------------------------", flush=True)
    #     run_attribution_patching(prompt, prompt_noisy_decoded, ["Wait", noisy_answer], example_id=0, at_token_index=j)
    #     break
    # You might want to add logic here similar to the previous loop,
    # e.g., calling run_attribution_patching if a condition is met.
    if "Wait" not in noisy_answer:
        print(f"Found the answer at step j={j}")
        run_attribution_patching(prompt, prompt_noisy_decoded, ["Wait", noisy_answer], example_id=2, at_token_index=j) # Using example_id=1 to differentiate