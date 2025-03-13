# LLM4Onto: Ontology Quality Evaluation and Generation

LLM4Onto is a framework designed to **evaluate ontology quality** and **generate ontologies using Large Language Models (LLMs)**. The ontology generation process expands upon an **input prompt written in an ontology format**, ensuring that the generated content follows a structured semantic representation. The repository focuses on two main aspects:

1. **Quality Metrics**: Analyzing and selecting high-quality ontologies.
2. **Ontology Generation**: Using a fine-tuned LLM to expand or generate ontologies, evaluated manually.

![LLM4Onto Methodology](image/methodology.jpg)

---

## 🛠️ Setup and Usage

### **1️⃣ Quality Metrics Computation**
To compute quality metrics for ontologies:
- Start a Docker container using:
  ```sh
  docker compose -f docker-compose-data-prep.yml up -d
  ```
- Execute the quality metrics script inside the container:
  ```sh
  python quality_metrics.py
  ```
- This generates an Excel file containing ontology quality scores.

### **2️⃣ Ontology Generation**
To generate ontologies using a fine-tuned LLM:
- Start the model Docker container with GPU support:
  ```sh
  docker compose -f docker-compose-model.yml up -d
  ```
- Run the ontology generation script:
  ```sh
  python prompt_gen.py
  ```
- **Input**: `prompts.json` defines ontology fragments that guide the model.
- **Output**: Generated ontologies are stored in the `outputs` directory.

### **3️⃣ Benchmark Evaluation**
- The generated ontologies are evaluated using both **manual review** and **benchmark comparisons**.
- Evaluation results are stored in the `results` folder.

---

## 📂 Repository Structure
```
├── data/                # Example ontologies for testing
├── outputs/             # Generated ontologies and quality metrics
├── results/             # Processed metrics and evaluations
├── scripts/             # Core processing scripts
├── docker/              # Docker configurations
├── README.md            # General documentation (this file)
```

---

## 🔍 Additional Notes
- **Three example ontologies** are provided in `data/`, with corresponding outputs.
- The `results/` folder contains **processed metrics from the DBpedia dataset**, generated ontologies, and their evaluations.
- Each section has its own **detailed README** with further instructions.

