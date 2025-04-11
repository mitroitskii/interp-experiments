# %% Imports
import torch as th
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from datasets import load_from_disk, load_dataset
from tqdm import tqdm


# %% Load models and tokenizers

def load_models():

    ft_model = AutoModelForCausalLM.from_pretrained(
        "agentica-org/DeepScaleR-1.5B-Preview",
        device_map="cuda:0", 
        torch_dtype=th.bfloat16
    )
    ft_tokenizer = AutoTokenizer.from_pretrained(
        "agentica-org/DeepScaleR-1.5B-Preview", padding_side = 'left')

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        device_map="cuda:1",
        torch_dtype=th.bfloat16
    )
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    return base_model, ft_model, base_tokenizer, ft_tokenizer

# %% Load dataset

def load_dataset_hf():
    return load_dataset('mitroitskii/OpenR1-Math-220k-formatted')['test']

# %% Evaluate tokenizer performance

def analyze_tokenizer(model, eval_tokenizer, dataset, num_samples=3, with_think_tag = True):
    """Analyze model performance using the given tokenizer
    
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
        
        text = [dataset[idx]['message'][0]]
        
        # 1. Apply chat template (still returning string)
        chat_text = eval_tokenizer.apply_chat_template(text,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=32768,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
        #chat_text = eval_tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        if with_think_tag:
            chat_text = text[0]["content"] + "<think>/n"
        else:
            chat_text = text[0]["content"]
        
        # 2. Tokenize the result
        chat_inputs = eval_tokenizer(chat_text, return_tensors="pt").to(device)
        
        # 3. Generate output
        generated_ids = model.generate(**chat_inputs, temperature=0.6, top_k=1, top_p=0.95, max_new_tokens=32768)
        output_text = eval_tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        if idx < 3:
            print("Example No.", idx)
            print(chat_text)
            print("------------------------------------------------------")
            print(output_text)
            print("======================================================")
        print("======================================================")

print("Loading models...")
base_model, ft_model, base_tokenizer, ft_tokenizer = load_models()

print("Loading dataset...")
dataset = load_dataset_hf()

print("\nRunning tokenizer compatibility tests...")

print("\nBase Model with Base Tokenizer...")
base_with_base_tokenizer = analyze_tokenizer(base_model, base_tokenizer, dataset, with_think_tag = False)

print("\nBase Model with FT Tokenizer...")
base_with_ft_tokenizer = analyze_tokenizer(base_model, ft_tokenizer, dataset, with_think_tag = False)

print("\nFT Model with FT Tokenizer...")
ft_with_ft_tokenizer = analyze_tokenizer(ft_model, ft_tokenizer, dataset, with_think_tag = False)

print("\nFT Model with Base Tokenizer...")
ft_with_base_tokenizer = analyze_tokenizer(ft_model, base_tokenizer, dataset, with_think_tag = False)