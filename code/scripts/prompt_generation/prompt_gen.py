import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global Configuration
MODEL_PATH = "meta-llama/Llama-3.2-1B" #Local or HuggingFace
PROMPTS_FILE = "prompts.json"  # JSON file storing unique prompts
OUTPUT_FILE = "/app/outputs/generated_ontologies.json"
GPU_ID = 0
MAX_LENGTH = 450
TOKEN = "your_token" # Can be anything if the model is loaded locally


def load_prompts(file_path):
    """
    Load prompts from an external JSON file.

    :param file_path: Path to the JSON file containing prompts.
    :return: Dictionary with prompt keys, their sources, and texts.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def load_model():
    """
    Load the language model and tokenizer. 
    Automatically assigns the device (GPU/CPU).

    :return: tokenizer, model, device
    """
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() and torch.cuda.device_count() > GPU_ID else "cpu")
    print(f"Using {device} for text generation.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,use_auth_token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,use_auth_token=TOKEN).to(device)

    return tokenizer, model, device


def generate_text(model, tokenizer, device, prompt_text):
    """
    Generate text using the model based on a given prompt.

    :param model: The pre-trained language model.
    :param tokenizer: Tokenizer for text processing.
    :param device: The device where the model is running (CPU/GPU).
    :param prompt_text: Input prompt for text generation.
    :return: Generated text as a string.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=MAX_LENGTH, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    """
    Main function to load prompts, generate text, and store results in a JSON file.
    """
    # Load prompts from the external JSON file
    prompts = load_prompts(PROMPTS_FILE)

    # Load existing results if available
    try:
        with open(OUTPUT_FILE, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        results = {}

    # Load model and tokenizer
    tokenizer, model, device = load_model()

    # Process each prompt
    for prompt_key, prompt_data in prompts.items():
        source = prompt_data["source"]  # Extract source
        prompt_text = prompt_data["text"]  # Extract actual text

        for i in range(6):  # Each prompt is used 6 times
            prompt_id = f"{prompt_key}_{i}"

            if prompt_id not in results:  # Avoid redundant generation
                generated_text = generate_text(model, tokenizer, device, prompt_text)
                
                # Store results with source information
                results[prompt_id] = {
                    "source": source,
                    "generated_text": generated_text
                }

                print(f"Generated text ({prompt_id} from {source}):", generated_text)

                # Save results after each iteration to prevent data loss
                with open(OUTPUT_FILE, "w") as file:
                    json.dump(results, file, ensure_ascii=False, indent=4)


# Execute the script
if __name__ == "__main__":
    main()
