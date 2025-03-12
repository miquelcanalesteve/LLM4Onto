import ast
import os
import pandas as pd
from collections import defaultdict

def load_data(filepath):
    """
    Loads the ontology metrics data from an Excel file.
    
    Args:
        filepath (str): Path to the Excel file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_excel(filepath)

def process_languages(data):
    """
    Processes the language data by filtering out empty keys,
    summing occurrences, and calculating total frequencies.

    Args:
        data (pd.DataFrame): The dataset containing language metrics.

    Returns:
        dict: Dictionary with language frequencies in percentage.
    """
    language_sums = defaultdict(float)
    total_sum = 0  # Sum of all language values

    for index in range(len(data)):
        # Convert the string representation of dictionary into an actual dictionary
        data.at[index, "Languages"] = ast.literal_eval(data.at[index, "Languages"])

        # Remove empty language keys ('')
        data.at[index, "Languages"] = {
            lang: value for lang, value in data.at[index, "Languages"].items() if lang
        }

        # Sum the values of remaining languages
        for lang, value in data.at[index, "Languages"].items():
            language_sums[lang] += value
            total_sum += value

    # Compute language frequencies
    language_frequencies = {
        lang: (value / total_sum) * 100 for lang, value in language_sums.items()
    }

    # Sort by frequency in descending order
    return dict(sorted(language_frequencies.items(), key=lambda x: x[1], reverse=True))

def save_frequencies_to_file(language_frequencies, output_dir):
    """
    Saves the language frequencies to a text file in the outputs directory.

    Args:
        language_frequencies (dict): Dictionary of language frequencies.
        output_dir (str): Path to the directory where the file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_file = os.path.join(output_dir, "language_frequencies.txt")

    with open(output_file, "w") as f:
        f.write("Language Frequency Distribution (%):\n")
        for lang, freq in language_frequencies.items():
            f.write(f"{lang}: {freq:.2f}%\n")
    
    print(f"Results saved to: {output_file}")

def main():
    """
    Main function to load data, process language frequencies, and save results.
    """
    filepath = "./../../outputs/ontology_metrics_lang.xlsx"
    output_dir = "./../../outputs"
    
    data = load_data(filepath)
    language_frequencies = process_languages(data)
    
    save_frequencies_to_file(language_frequencies, output_dir)

if __name__ == "__main__":
    main()
