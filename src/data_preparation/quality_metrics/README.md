# Ontology Quality Metrics and Dataset Selection

## Overview

This repository contains scripts to compute ontology quality metrics and facilitate dataset selection for continual pretraining. The methodology is based on three key factors:

1. **Average Subclasses per Class (SC)** – Measures the hierarchical depth of an ontology.
2. **Average Non-Taxonomic Relations per Class (NTR)** – Represents the density of non-taxonomic relationships.
3. **Property Density (PD)** – Indicates the knowledge richness of the ontology.

These metrics are computed using the `rdflib` Python library, while token counts are obtained using the `transformers` library. The scripts process ontology files in Turtle format (`.ttl`) and generate structured output in an Excel file.

## Scripts

### `quality_metrics.py`

This script processes ontologies and computes key quality metrics.

#### **Functionality:**
- Identifies and counts ontology classes.
- Extracts object properties, data properties, and annotation properties.
- Computes:
  - Property Density (PD)
  - Non-Taxonomic Relations per Class (NTR)
  - Subclasses per Class (SC)
- Counts the total number of triples and tokens.
- Outputs results in an Excel file for further analysis.

#### **Input:**
- A folder containing ontology files in Turtle (`.ttl`) format.

#### **Output:**
- An Excel file (`ontology_metrics.xlsx`) stored in the `results` folder, containing:
  - **File Name** – Ontology file name.
  - **Total Tokens** – Number of tokens in the ontology file.
  - **Total Triples** – Number of RDF triples.
  - **Property Density (PD)** – Average number of properties per class.
  - **Non-Taxonomic Relations per Class (NTR)** – Average non-taxonomic relationships per class.
  - **Subclasses per Class (SC)** – Average number of subclasses per class.

## Dataset Selection Process

Once `quality_metrics.py` generates `ontology_metrics.xlsx`, further processing is done manually in Excel. The file, along with its formulas, is stored inside the `results` folder.

1. **Normalization and Quality Score Calculation:**
   - Min-max normalization is applied to PD, SC, and NTR.
   - A **Quality Score (QS)** is computed as the sum of normalized values.
   - The contribution of each ontology to the total token count is considered. As ontologies are ordered by QS, a cumulative token count is maintained, allowing users to track how much of the total corpus is selected at any given point.

2. **Ontology Selection:**
   - The ontologies are sorted in descending order based on QS.
   - The user can then choose the ontologies to include in the training dataset according to their criteria, ensuring control over dataset size and quality.

## Summary
- `quality_metrics.py` extracts only the necessary quality metrics (PD, NTR, and SC) from ontologies and saves the results in an Excel file in the `results` folder.
- Metrics are normalized and a Quality Score is computed in Excel.
- `ontology_metrics.xlsx`, including its formulas, is stored in the `results` folder.
- The cumulative token contribution of selected ontologies is tracked, helping users understand what portion of the corpus they are including.
- Ontologies are ranked by QS, allowing the user to manually select those to include in the training dataset.
- This structured approach ensures a systematic and efficient selection of high-quality ontologies for continual pretraining.

