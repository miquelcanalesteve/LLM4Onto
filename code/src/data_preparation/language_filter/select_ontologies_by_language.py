import ast
import pandas as pd

def load_data(filepath):
    """
    Loads the ontology metrics data from an Excel file.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_excel(filepath)

def find_files_without_english(data):
    """
    Identifies files that do not contain English ("en") in their language distribution.

    Args:
        data (pd.DataFrame): The dataset containing language metrics.

    Returns:
        list: A list of file names that do not include English.
    """
    files_without_en = []

    for index in range(len(data)):
        data.at[index, "Languages"] = ast.literal_eval(data.at[index, "Languages"])
        
        # Check if "en" is not in the language dictionary
        if "en" not in data.at[index, "Languages"]:
            files_without_en.append(data.at[index, "File Name"])

    return files_without_en

def save_results(file_list, output_file="./../../outputs/files_without_english.txt"):
    """
    Saves the list of files without English to a text file.

    Args:
        file_list (list): List of file names.
        output_file (str): Path to the output text file.
    """
    with open(output_file, "w") as f:
        for file in file_list:
            f.write(file + "\n")
    print(f"File names have been saved in '{output_file}'.")

def main():
    """
    Main function to load data, find files without English, and display/save results.
    """
    filepath = "./../../outputs/ontology_metrics_lang.xlsx"
    data = load_data(filepath)
    files_without_en = find_files_without_english(data)

    # Display results
    print("Files without English:")
    for file in files_without_en:
        print(file)

    print(f"\nTotal files without English: {len(files_without_en)}")

    # Save results to a text file
    save_results(files_without_en)

if __name__ == "__main__":
    main()
