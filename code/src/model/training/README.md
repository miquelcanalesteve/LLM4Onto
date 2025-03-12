# 🎮 Ontology-Based Model Training Module

This module is responsible for **tokenizing, training, and converting** a Llama 3.2-1B model for **ontology-based continual learning**. The module follows a structured pipeline with multiple scripts to ensure **efficient dataset preparation, model training, and format conversion**.

---

## 🚀 Overview

The training process consists of **three main steps**:

1. **Tokenization**: Prepares ontology fragments for model training.
2. **Continual Training**: Performs continual learning using the tokenized dataset.
3. **Model Conversion**: Converts the trained model to Hugging Face format for easier deployment.

---

## 🛠️ Installation & Setup

### 1️⃣ Install Dependencies

Ensure you have **Python ≥3.8** and install the required libraries:
```sh
pip install torch transformers datasets lightning
```

### 2️⃣ Prepare the Model & Dataset

- Place your **pre-trained Llama 3.2-1B model** in the following directory:
  ```
  /workspace/NAS/GPLSI/llm-train-tokenizer-custom-dataset-main/modelos/Llama-3.2-1B_df_calidad_alta/epoch_4
  ```
- Ensure the **raw ontology dataset** is located inside a folder **one level below** the `input_dir`.
  
---

## 🎯 Execution Order

### **1. Tokenization** (Prepares Data for Training)

```sh
python tokenization_main.py
```

This script tokenizes **TTL ontology files** and prepares them for continual pretraining.

#### 🔄 Input:
- `tokenization_arguments.py` provides configurable arguments.
- `input_dir` must point to a **parent folder** containing the ontology files.

#### 📤 Output:
- **Splitted ontology dataset** (if files exceed size limits).
- **Arrow dataset** stored in Hugging Face format.
- **Tokenized dataset** with overlapping tokenized sequences.

---

### **2. Continual Training** (Fine-tunes Llama 3.2-1B on Ontology Data)

```sh
python continual_ddp.py
```

This script runs **distributed continual pretraining** with configurable training parameters.

#### 🔄 Input:
- `llama_3.2_1b_semantic.json` (defines **base model path**, **training data**, and **hyperparameters**).
- `utils/` folder (contains helper functions for training).

#### ⚙️ Key Configurable Parameters:
| Parameter | Description | Default |
|-----------|------------|---------|
| `model_name` | Base model used for training | `meta-llama/Llama-3.2-1B` |
| `train_data_dir` | Path to tokenized dataset | `/workspace/data/tokenized_df_calidad_alta` |
| `number_epochs` | Number of training epochs | `2` |
| `batch_size` | Batch size per device | `1` |
| `gradient_accumulation_steps` | Steps before optimizer update | `64` |
| `lr` | Learning rate | `2e-5` |
| `weight_decay` | Weight decay for optimizer | `1e-1` |
| `precision` | Training precision format | `bf16-true` |
| `max_step` | Maximum training steps | `8192` |
| `warmup_steps` | Steps before LR scheduling starts | `0` |
| `save_step_interval` | Checkpoint saving interval | `1000` |
| `eval_step_interval` | Validation interval | `500` |
| `grad_clip` | Gradient clipping threshold | `1.0` |
| `log_step_interval` | Logging interval for monitoring | `1` |

#### 📤 Output:
- Model checkpoints saved in `/data/meta-llama/Llama-3.2-1B_df_calidad_alta_e45/`.
- Training logs with **loss values** and **performance metrics**.

---

### **3. Model Conversion** (Converts Model to Hugging Face Format)

```sh
python convert_fabric_to_hf_models.py
```

This script **converts the trained model** to Hugging Face format for compatibility with **generation scripts**.

#### 🔄 Input:
- `checkpoint_path` (path to trained model checkpoint).
- `config_model.json` (defines model architecture and tokenizer settings).

#### 📤 Output:
- Hugging Face **compatible model** stored in `output_dir`.

Example default output path:
```
/workspace/data/modelos/Llama-3.2-1B_df_calidad_alta/epoch_4
```

---

## 📊 Performance Considerations

- ✅ **Multi-GPU support** for efficient training.
- ✅ **Incremental checkpoint saving** to prevent data loss.
- ✅ **Hugging Face format conversion** for easy deployment.

---

## 🤝 Contributions

If you encounter any issues or have suggestions for improvement, feel free to submit a **pull request** or open an **issue**.

📧 **Contact**: your.email@example.com  
👨‍💻 **Author**: Your Name  

---

## 📚 License

MIT License © 2025 Your Name

