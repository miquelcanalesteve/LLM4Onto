
# Data Preparation Pipeline

This repository contains scripts for preparing a high-quality ontology dataset for continual pretraining. The process consists of two main stages: **language filtering** and **quality-based selection**.

## Workflow

### 1. Language Filtering
The first step ensures that only ontologies containing English literals are included in the dataset.

#### **Scripts:**
- `identify_language.py`: Analyzes each ontology and detects the percentage of different languages in its literals. Outputs an Excel file (`ontology_metrics_lang.xlsx`).
- `select_ontologies_by_language.py`: Filters out ontologies that do not contain English literals and saves a list of excluded files (`files_without_english.txt`).
- `language_frequencies.py`: Computes the overall language distribution and stores it in `language_frequencies.txt`.

After running these scripts, only ontologies with English content are retained for further processing.

---

### 2. Quality-Based Ontology Selection
Once the dataset is filtered by language, the next step is to compute quality metrics and select the best ontologies for pretraining.

#### **Scripts:**
- `quality_metrics.py`: Computes ontology quality metrics, including:
  - Property Density (PD)
  - Non-Taxonomic Relations per Class (NTR)
  - Subclasses per Class (SC)
  - Quality Score (QS), which ranks ontologies based on normalized metrics.
  - Cumulative token count and percentage.
  - Saves results in `ontology_metrics.xlsx`.
- **Manual threshold selection**: The user selects a Quality Score threshold to determine which ontologies to include in the final dataset.
- `select_ontologies.py`: Filters and copies ontologies that exceed the chosen QS threshold into the final dataset folder.

After this stage, the dataset is ready for tokenization.

---

## Final Output
After following the pipeline, we obtain a curated folder of high-quality ontologies that meet both **language** and **quality** criteria, ensuring an optimal dataset for continual pretraining.

