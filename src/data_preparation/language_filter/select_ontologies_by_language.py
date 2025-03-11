import ast
import pandas as pd

data = pd.read_excel("ontology_metrics_lang.xlsx")
# List to store file names without English
files_without_en = []

for k in range(len(data)):
    data.at[k, "Languages"] = ast.literal_eval(data.at[k, "Languages"])
    file_name = data.at[k, "File Name"]

    # Check if "en" is not present in the language dictionary
    if "en" not in data.at[k, "Languages"]:
        files_without_en.append(data.at[k, "File Name"])

# Display file names without English
print("Files without English:")
for file in files_without_en:
    print(file)

# Save to a TXT file if needed
with open("files_without_english.txt", "w") as f:
    for file in files_without_en:
        f.write(file + "\n")

print(f"\nTotal files without English: {len(files_without_en)}")
print("The file names have been saved in 'files_without_english.txt'.")
