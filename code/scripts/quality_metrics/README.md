# Ontology Quality Metrics and Dataset Selection

## Overview

This repository contains scripts for computing ontology quality metrics and facilitating dataset selection for continual pretraining. The methodology is based on three key factors:

1. **Average Subclasses per Class (SC)** – Measures the hierarchical depth of an ontology.
2. **Average Non-Taxonomic Relations per Class (NTR)** – Represents the density of non-taxonomic relationships.
3. **Property Density (PD)** – Indicates the knowledge richness of the ontology.

These metrics are computed using the `rdflib` Python library, while token counts are obtained using the `transformers` library. The scripts process ontology files in Turtle format (`.ttl`) and generate structured output in an Excel file.

## Script

### `quality_metrics.py`

This script processes ontologies and computes key quality metrics.

#### **Functionality:**
- Identifies and counts ontology classes.
- Extracts object properties, data properties, and annotation properties.
- Computes:
  - Property Density (PD)
    $$
    PD = \frac{\sum_{i=1}^{c}{n_{att}(i) + r_{not}(i)}}{c}
    $$
  - Non-Taxonomic Relations per Class (NTR)
    $$
    NTR = \frac{\sum_{i=1}^{c}{r_{not}(i)}}{c}
    $$
  - Subclasses per Class (SC)
    $$
    SC = \frac{\sum_{i=1}^{c}{s(i)}}{c}
    $$
- Counts the total number of triples and tokens.
- Normalizes the computed metrics using Min-Max scaling.
- Computes a **Quality Score (QS)** as the sum of normalized values.
- Tracks the cumulative token count percentage.
- Outputs results in an Excel file for further analysis.

Where:
- $ c $ represents the number of classes.
- $ n_{\text{att}}(i) $ corresponds to the number of data properties or attributes of a class.
- $ r_{\text{not}}(i) $ denotes the number of non-taxonomic relationships of a class.
- $ s(i) $ is the number of subclasses of a class.



#### **Input:**
- A folder containing ontology files in Turtle (`.ttl`) format.

#### **Output:**
- An Excel file (`ontology_metrics.xlsx`) stored in the `outputs` folder, containing:
  - **File Name** – Ontology file name.
  - **Total Tokens** – Number of tokens in the ontology file.
  - **Total Triples** – Number of RDF triples.
  - **Property Density (PD)** – Average number of properties per class.
  - **Non-Taxonomic Relations per Class (NTR)** – Average non-taxonomic relationships per class.
  - **Subclasses per Class (SC)** – Average number of subclasses per class.
  - **Normalized Property Density (norm PD)** – Min-max normalized value of PD.
  - **Normalized Non-Taxonomic Relations (norm NTR)** – Min-max normalized value of NTR.
  - **Normalized Subclasses per Class (norm SC)** – Min-max normalized value of SC.
  - **Quality Score (QS)** – Sum of the three normalized values.
  - **Token Count Accumulation** – Cumulative sum of tokens ordered by QS.
  - **Percentage of Token Count Accumulation** – Percentage of total token count accumulated.

---

## Dataset Selection Process

Once `quality_metrics.py` generates `ontology_metrics.xlsx`, the selection of ontologies proceeds as follows:

**Ontology Selection:**
   - The ontologies are sorted in descending order based on QS.
   - The user can then choose the ontologies to include in the training dataset according to their criteria, ensuring control over dataset size and quality.
   - The script `select_ontologies.py` automates the selection process based on a predefined QS threshold.

## Summary
- `quality_metrics.py` extracts and normalizes ontology quality metrics, saving the results in an Excel file in the `outputs` folder.
- The cumulative token contribution of selected ontologies is tracked, helping users understand what portion of the corpus they are including.
- Ontologies are ranked by QS, allowing the user to manually or automatically select those to include in the training dataset.
- This structured approach ensures a systematic and efficient selection of high-quality ontologies for continual pretraining.