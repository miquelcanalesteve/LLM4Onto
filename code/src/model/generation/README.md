# 🧠 Ontology-Based Text Generation Module

This module is responsible for generating text completions based on **ontology fragments** using a **pre-trained Llama 3.2-1B** model. The generated texts help evaluate the model's generalization ability on unseen ontology fragments.

---

## 🚀 Overview

This module processes **ontology fragments** sourced from well-established repositories, including:

- **AGRO** → Agriculture  
- **EDAM** → Biology & Bioinformatics  
- **MDS** → Materials Science  
- **SWEET** → Spatial & Environmental Sciences  

Each fragment is **randomly selected**, limited to **150 tokens**, and used as input for the model. To ensure diverse outputs, each fragment is processed **six times**, generating different completions.

This script is designed for **efficient batch processing**, with incremental saving to prevent data loss.

---

## 🛠️ Installation & Setup

### 1️⃣ Install Dependencies

Ensure you have **Python ≥3.8** and install the required libraries:
```sh
pip install torch transformers
```

### 2️⃣ Prepare the Model

Download or place your **pre-trained Llama 3.2-1B model** in the following directory:
```
/workspace/NAS/GPLSI/llm-train-tokenizer-custom-dataset-main/modelos/Llama-3.2-1B_df_calidad_alta/epoch_4
```

### 3️⃣ Run the Script

Execute the script to generate text:
```sh
python generate_text.py
```

---

## 🎯 How It Works

### 💚 Input: `prompts.json`

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
3️⃣ The generated texts are **incrementally saved** in `generated_texts.json` to prevent data loss.  

### 📤 Output: `generated_texts.json`

After running the script, the generated texts are stored in `generated_texts.json` in the following format:

```json
{
    "citation_0": {
        "source": "edam",
        "generated_text": "Generated text for citation..."
    },
    "has_identifier_1": {
        "source": "edam",
        "generated_text": "Generated text for has_identifier..."
    }
}
```

---

## ⚙️ Configurable Parameters

Modify these parameters in `generate_text.py` to adjust settings:

| Parameter       | Description | Default Value |
|----------------|-------------|--------------|
| `MODEL_PATH`   | Path to the pre-trained model. | `"/workspace/.../epoch_4"` |
| `PROMPTS_FILE` | JSON file containing ontology fragments. | `"prompts.json"` |
| `OUTPUT_FILE`  | JSON file where generated texts are saved. | `"generated_texts.json"` |
| `GPU_ID`       | The GPU to use. Falls back to CPU if unavailable. | `4` |
| `MAX_LENGTH`   | Maximum token length for generation. | `450` |
| `do_sample`    | Enables sampling for diverse outputs. | `True` |
| `top_k`        | Restricts token sampling to top-k probabilities. | `50` |
| `top_p`        | Nucleus sampling threshold. | `0.95` |
| `temperature`  | Adjusts randomness in sampling. | `0.7` |

If **deterministic output** is required, set:
```python
do_sample = False
```
However, **this increases repetition errors**, so the default settings are recommended.

---

## 📊 Performance Considerations

- ✅ **Uses GPU if available** for faster processing.
- ✅ **Incremental saving** prevents loss of generated outputs.
- ✅ **Multiple generations (6x per fragment)** ensure diverse outputs.

---

## 🤝 Contributions

If you encounter any issues or have suggestions for improvement, feel free to submit a **pull request** or open an **issue**.

📧 **Contact**: your.email@example.com  
👨‍💻 **Author**: Your Name  

---

## 📚 License

MIT License © 2025 Your Name

