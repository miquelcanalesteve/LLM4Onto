import pandas as pd
import os 
from rdflib import Graph, Literal
from langdetect import detect
from collections import Counter


def extract_textual_metrics(g):
    """
    Extracts textual metrics from the dataset, focusing on literal objects.

    :param g: RDF graph.
    :return: Dictionary with textual metrics.
    """
    literals = [o for s, p, o in g.triples((None, None, None)) if isinstance(o, Literal)]
    
    language=[]
    for literal in literals:
        try:
            language.append(detect(literal))
        
        except Exception as e:
            if "No features in text." not in str(e):  # Only shows an error if it's a different exception. 
                                                      # This one means that no language was detected
                print("Exception:", e)
            language.append('')

    total_elements  = len(language)
    language_counts  = Counter(language)

    language_percentages  = {lang: count / total_elements  for lang, count in language_counts .items()}
    return language_percentages 

def process_ttl_file(file_path):
    """
    Process a single TTL file and calculate the metrics.

    :param file_path: Path to the TTL file.
    :return: Dictionary with calculated metrics for the file.
    """
    try:
        # Load the graph and calculate metrics
        g = Graph()
        g.parse(file_path, format="turtle")

        # Extract textual metrics from literals
        languages = extract_textual_metrics(g)

        return {
            "File Name": os.path.basename(file_path),
            "Languages":languages}

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":
    folder_path = "./../../data/full_dataset" # Folder containing the TTL files
    output_excel = "./../../outputs/ontology_metrics_lang.xlsx"

    # Initialize an empty DataFrame
    columns = [
        "File Name", "Languages"
    ]
    df = pd.DataFrame(columns=columns)

    # Process each TTL file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".ttl"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")
            
            # Process the file and calculate metrics
            literals = process_ttl_file(file_path)
            
            df = pd.concat([df, pd.DataFrame([literals])], ignore_index=True)
                
            # Save the updated DataFrame to Excel
            df.to_excel(output_excel, index=False)
            print(f"Metrics for {file_name} saved to {output_excel}")

    print(f"Processing complete. Final results saved to {output_excel}")