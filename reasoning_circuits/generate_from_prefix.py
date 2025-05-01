# NOTE experiment in progress
# FIXME do imports; connect with the inputs

# %%

# Find the token ID for "Wait" (assuming it's a single token, might need adjustment if not)
# Need to be careful about spaces, e.g., " Wait" vs "Wait"
wait_token_str = " Wait"  # Common tokenization includes space before word
wait_token_id = tokenizer.encode(wait_token_str, add_special_tokens=False)

if len(wait_token_id) > 1:
    print(
        f"Warning: '{wait_token_str}' is tokenized into multiple IDs: {wait_token_id}. Using the first one: {wait_token_id[0]}")
wait_token_id = wait_token_id[0]

print(wait_token_id)
print(tokenizer.decode(wait_token_id))

# %%
sampling_outputs_temp_06_tokens = results["Sampling (temp=0.6)"]["tokens"]
sampling_response_temp_06 = results["Sampling (temp=0.6)"]["text"]

# Find the index of the "Wait" token in the generated sequence
original_output_tokens = sampling_outputs_temp_06_tokens[0]
wait_token_indices = (original_output_tokens ==
                      wait_token_id).nonzero(as_tuple=True)[0]

if len(wait_token_indices) == 0:
    print(
        f"Error: Token '{wait_token_str}' (ID: {wait_token_id}) not found in the generated output.")
    # Handle error appropriately, maybe try without the leading space or stop
else:
    # Use the first occurrence if multiple exist
    wait_token_index = wait_token_indices[0].item()
    print(
        f"Found '{wait_token_str}' token (ID: {wait_token_id}) at index {wait_token_index}.")

    # Truncate the sequence up to the "Wait" token
    truncated_input_ids = original_output_tokens[:wait_token_index].unsqueeze(
        0)  # Add batch dimension

    # Reset the seed to ensure reproducibility for the continuation
    print(f"\nResetting seed to {seed} and generating continuation...")
    set_seed(seed)

    # Generate the continuation with the same parameters
    continuation_outputs = model.generate(
        truncated_input_ids,
        max_new_tokens=1000,  # Allow generating enough tokens for comparison
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
    )

    # Decode the new full response
    continuation_response = tokenizer.decode(
        continuation_outputs[0], skip_special_tokens=True)

    # Extract the part after "Wait" from both original and continuation
    original_continuation = sampling_response_temp_06.split(wait_token_str, 1)[
        1] if wait_token_str in sampling_response_temp_06 else "[Wait token not found in string]"
    new_continuation = continuation_response.split(wait_token_str, 1)[
        1] if wait_token_str in continuation_response else "[Wait token not found in string]"

    print("\nOriginal response (temp 0.6):")
    print(sampling_response_temp_06)
    print("\nResponse generated from truncated input (temp 0.6):")
    print(continuation_response)

    # Compare the continuations
    print("\n--- Comparison ---")
    print(
        f"Original continuation after '{wait_token_str}':\n{original_continuation}")
    print(f"\nNew continuation after '{wait_token_str}':\n{new_continuation}")

    if original_continuation == new_continuation:
        print("\nResult: The continuation generated from the truncated input MATCHES the original continuation.")
    else:
        print("\nResult: The continuation generated from the truncated input DOES NOT MATCH the original continuation.")

# %
