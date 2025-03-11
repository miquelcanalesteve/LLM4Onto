import os
import pandas as pd
from transformers import AutoTokenizer

def count_tokens(text):
    # Load the tokenizer for Llama3
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        use_auth_token="your_token"
    )
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def count_tokens_in_folder(folder_path, output_csv_path):
    total_tokens = 0
    data = []

    for file in os.listdir(folder_path):
        if file.endswith(".ttl"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                num_tokens = count_tokens(content)
                total_tokens += num_tokens
                print(f"File: {file} - Tokens: {num_tokens}")
                data.append({"File Name": file, "Token Count": num_tokens})

    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(data)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8')

    print(f"\nTotal tokens in the folder '{folder_path}': {total_tokens}")
    print(f"Results have been saved to '{output_csv_path}'")

if __name__ == "__main__":
    folder_path = "/workspace/NAS/mikel/ontologias/ontologias_dataset_crudo/dbpedia-archivo/ttl/ttl"  # Change this to the desired path
    output_csv_path = "token_results.csv"  # CSV file where results will be saved
    count_tokens_in_folder(folder_path, output_csv_path)
