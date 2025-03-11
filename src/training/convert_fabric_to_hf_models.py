import argparse
import os
from transformers import AutoTokenizer
import lightning as L
from models.models_class import FabricGeneration
import json


def main(args):
    # Load the model config
    with open(os.path.join(os.getcwd(), args.config_model)) as f:
        config_model = json.load(f)

    checkpoint_path = os.path.join(os.getcwd(), args.checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(config_model['model_name'],token="your_token")
    tokenizer.pad_token = tokenizer.eos_token
    fabric = L.Fabric(accelerator=args.accelerator, devices=args.devices, strategy=args.strategy, precision=args.precision)
    fabric.launch()
    model = FabricGeneration(config_model)
    state = {"model": model}
    print(checkpoint_path)
    fabric.load(checkpoint_path, state)
    model = fabric.setup(model)

    outputs_dir = os.path.join(os.getcwd(), args.output_dir)
    # Check if the output directory exists, else create it
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    model.model.save_pretrained(outputs_dir)
    tokenizer.save_pretrained(outputs_dir)
    
    """ os.mkdir(outputs_dir)
    model.model.save_pretrained(outputs_dir)
    tokenizer.save_pretrained(outputs_dir) """


output_dir = "/workspace/data/modelos/Llama-3.2-1B_df_calidad_alta/epoch_4"#"/code/LLAMA/modelos/llama3-8B-instruct-semantic-ttl/epoch_1/"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_model", default="/workspace/LMM_continual_fine_tunning/data/Llama/llama_3.2_1b_model_semantic_base.json")#"/code/data/Llama/llama_model_semantic_instruct.json")
    parser.add_argument("--checkpoint_path", default="/data/meta-llama/Llama-3.2-1B_df_calidad_alta_e45/iter-065890-ckpt.pth")     
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--output_dir", default=output_dir)   # /raid/gplsi/NAS/GPLSI/LLAMA/modelos/
    args = parser.parse_args()
    main(args)
