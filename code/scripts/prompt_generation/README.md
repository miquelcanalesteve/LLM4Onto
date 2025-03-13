# Ontology Generation Module

This module generates **ontology fragments** based on a given **input prompt** using a **pre-trained Llama 3.2-1B** model. The output is stored in a structured JSON format for further processing or evaluation.

---

## Overview

This module processes **ontology fragments** sourced from well-established repositories, including:

- **AGRO** → Agriculture  
- **EDAM** → Biology & Bioinformatics  
- **MDS** → Materials Science  
- **SWEET** → Spatial & Environmental Sciences  

Each fragment is **randomly selected**, limited to **150 tokens**, and used as input for the model. To ensure diverse outputs, each fragment is processed **six times**, generating different ontology completions.

The script is designed for **efficient batch processing**, using a **GPU** when available. For small models like **Llama 3.2-1B**, this approach ensures fast execution.

---

## Installation & Setup

### 1️⃣ Execution Environment

This module is designed to run inside a **Docker container** using the provided `docker-compose.yml` file. All dependencies and necessary configurations (volumes, GPU access) are pre-defined.

To set up and start the container:
```sh
cd docker
docker-compose up --build -d
```

To access the running container:
```sh
docker exec -it docker-llm4onto-1 bash
```

If you want to run it outside the container, ensure you have **Python ≥3.8** and install the required libraries:
```sh
pip install torch transformers
```

### 2️⃣ Prepare the Model

You can specify the path where your model is located or use a model from Hugging Face Hub by setting its name in `MODEL_PATH`.

### 3️⃣ Run the Script

Inside the Docker container, execute:
```sh
python generate_text.py
```

---

## How It Works

### Input: `prompts.json`

This file contains ontology fragments along with their sources. Each entry has:
- A **text fragment** from an ontology.
- The **source repository** (e.g., `"edam"`, `"mds-onto"`).

Example:
```json
{
    "citation": {
        "source": "edam",
        "text": "###  http://edamontology.org/citation\n:citation rdf:type owl:AnnotationProperty ..."
    }
}
```

### 🔄 Processing Workflow

1️⃣ The script **loads prompts** from `prompts.json`.  
2️⃣ Each fragment is processed **six times**, generating **six different outputs**.  
3️⃣ The generated ontologies are **incrementally saved** in `generated_ontologies.json` to prevent data loss.  

### Output: `generated_ontologies.json`

After running the script, the generated ontologies are stored in `generated_ontologies.json` in the following format:

```json
{
    "citation_0": {
        "source": "edam",
        "generated_text": "Generated ontology for citation..."
    },
    "has_identifier_1": {
        "source": "edam",
        "generated_text": "Generated ontology for has_identifier..."
    }
}
```

A sample of the expected output is included in the `generated_ontologies.json` file.

---

## Performance Considerations

- ✅ **Uses GPU if available** for faster processing.  
- ✅ **Incremental saving** prevents loss of generated outputs.  
- ✅ **Multiple generations (6x per fragment)** ensure diverse ontology outputs.  

