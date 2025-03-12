import pandas as pd
import os
import shutil

# Configuration
EXCEL_FILE = "./../../outputs/ontology_metrics.xlsx"
QUALITY_THRESHOLD = 2.210636645962733
TTL_DIR = "./../../data/full_dataset"
DEST_DIR_QUALITY = "./../../data/selected_ontologies/selected_ontologies"


def load_data(file_path, sheet_name="Sheet1"):
    """
    Load the ontology metrics from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to read.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def filter_ontologies(df, threshold):
    """
    Filter the ontologies that have a quality score above the given threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing ontology metrics.
        threshold (float): Quality score threshold.
    
    Returns:
        set: Set of ontology file names that meet the quality criteria.
    """
    return set(df[df["Quality Score"] > threshold]["File Name"].dropna())


def copy_selected_files(source_dir, destination_dir, file_names):
    """
    Copy selected files from the source directory to the destination directory.
    
    Args:
        source_dir (str): Path to the source directory.
        destination_dir (str): Path to the destination directory.
        file_names (set): Set of file names to copy.
    """
    os.makedirs(destination_dir, exist_ok=True)  # Ensure destination directory exists

    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(destination_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)  # Preserve metadata
            print(f"Copied to high quality folder: {file_name}")
        else:
            print(f"Not found in source directory: {file_name}")

    print("Copy process completed.")


def main():
    """
    Main function to execute the ontology file selection and copying process.
    """
    df = load_data(EXCEL_FILE)
    selected_files = filter_ontologies(df, QUALITY_THRESHOLD)
    copy_selected_files(TTL_DIR, DEST_DIR_QUALITY, selected_files)


if __name__ == "__main__":
    main()
