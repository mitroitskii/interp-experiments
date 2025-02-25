# %% Imports
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm


# %% Load models and tokenizers

def load_models():

    ft_model = AutoModelForCausalLM.from_pretrained(
        "agentica-org/DeepScaleR-1.5B-Preview",
        device_map="cuda:0", 
        torch_dtype=th.bfloat16
    )
    ft_tokenizer = AutoTokenizer.from_pretrained(
        "agentica-org/DeepScaleR-1.5B-Preview")

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        device_map="cuda:1",
        torch_dtype=th.bfloat16
    )
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

    return base_model, ft_model, base_tokenizer, ft_tokenizer

# %% Load dataset

def load_dataset():
    return load_from_disk('~/.cache/huggingface/datasets/OpenR1-Math-220k-formatted')['test']

# %% Evaluate tokenizer performance

def evaluate_tokenizer_performance(model, eval_tokenizer, dataset, num_samples=1000):
    """Evaluate model performance using the given tokenizer
    
    Args:
        model: The model to evaluate
        eval_tokenizer: The tokenizer to test with this model
        dataset: Dataset containing text samples
        num_samples: Number of samples to evaluate
    """
    device = model.device
    total_loss = 0.0
    count = 0
    
    for idx in tqdm(range(min(num_samples, len(dataset)))):
        text = dataset[idx]['message_qwen1.5b']
        
        # Tokenize with the tokenizer we're evaluating
        inputs = eval_tokenizer(text, return_tensors="pt", 
                              truncation=True, max_length=512).to(device)
        
        # Skip if empty
        if inputs.input_ids.shape[1] == 0:
            continue
            
        # Forward pass
        with th.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            
        total_loss += loss.item()
        count += 1
    
    return total_loss / count  # Return average loss


# Add this to main() after loading models
def main():
    print("Loading models...")
    base_model, ft_model, base_tokenizer, ft_tokenizer = load_models()

    print("Loading dataset...")
    dataset = load_dataset()
    
    print("\nRunning tokenizer compatibility tests...")
    
    # Test base model with both tokenizers
    base_with_base_tokenizer = evaluate_tokenizer_performance(base_model, base_tokenizer, dataset)
    base_with_ft_tokenizer = evaluate_tokenizer_performance(base_model, ft_tokenizer, dataset)
    
    # Test ft model with both tokenizers
    ft_with_ft_tokenizer = evaluate_tokenizer_performance(ft_model, ft_tokenizer, dataset)
    ft_with_base_tokenizer = evaluate_tokenizer_performance(ft_model, base_tokenizer, dataset)
    
    print(f"\nBase model mean loss over 1000 samples:")
    print(f"With base tokenizer: {base_with_base_tokenizer:.4f}")
    print(f"With distilled tokenizer: {base_with_ft_tokenizer:.4f}")
    
    print(f"\nDistilled model mean loss over 1000 samples:")
    print(f"With distilled tokenizer: {ft_with_ft_tokenizer:.4f}")
    print(f"With base tokenizer: {ft_with_base_tokenizer:.4f}")

if __name__ == "__main__":
    main()
# %%
