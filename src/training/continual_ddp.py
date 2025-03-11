import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy, DeepSpeedStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
import tqdm
from tqdm.auto import tqdm, trange
import torch.nn as nn
import pickle as pkl
import torch.utils.checkpoint
from functools import partial

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually

from pytorch_lightning.loggers import WandbLogger
from utils.speed_monitor import SpeedMonitorFabric as Monitor
from utils.logger import step_csv_logger

# New importer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from datasets import load_from_disk
import os
import json
from models.models_class import FabricGeneration
# 1. Import a suiting wrapping policy from PyTorch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from lightning.fabric.strategies import DDPStrategy



checkpoint_path = "out/Llama"
config_model = "/../data/Llama/llama_3.2_1b_model_semantic_base.json"

# Load the model config
with open(os.getcwd() + config_model) as f:
    config_model = json.load(f)
#out_dir = Path("out") / config_model["model_name"]
out_dir = Path("/data") / config_model["model_name"]

min_lr = 2e-5
eval_iters = 1000000
micro_batch_size = config_model["batch_size"]


warmup_iters = config_model["warmup_steps"] * config_model["gradient_accumulation_steps"]
max_iters = config_model["max_step"] * config_model["gradient_accumulation_steps"]
lr_decay_iters = max_iters
log_iter_interval = config_model["log_step_interval"] * config_model["gradient_accumulation_steps"]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", config_model["model_name"], flush_logs_every_n_steps=log_iter_interval)
wandb_logger = WandbLogger(entity="gplsi_continual", project="fabric_llama", log_model="all")


def setup(
    devices: int = 4,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
) -> None:
    
    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = DDPStrategy(find_unused_parameters=False,gradient_as_bucket_view=True)

            """ strategy = DeepSpeedStrategy(
                zero_optimization=True,
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
                partition_activations=True,
                cpu_checkpointing=True,
            ) """

    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=config_model["precision"], loggers=[logger, wandb_logger])

    fabric.print(hparams)
    fabric.launch(main, config_model, resume)


def main(fabric, config_model, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Get the data
    dataset = load_from_disk(config_model["train_data_dir"])
    train_dataset = dataset["train"]
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    #subset = train_dataset.select(range(100))
    
    validation_dataset = dataset["valid"] #valid
    validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    #val_subset = validation_dataset.select(range(100))

    train_dataloader = DataLoader(train_dataset, batch_size=config_model["batch_size"], shuffle=True)  #train_dataset

    validation_dataloader = DataLoader(validation_dataset, batch_size=config_model["eval_batch_size"], shuffle=False) #validation_dataset

    if validation_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, validation_dataloader)

    # fabric.seed_everything(3407)  # used only in FSDP

    t0 = time.perf_counter()
    """ model = FabricGeneration(config_model)
    model.to(fabric.device) """
    with fabric.init_module():
        model = FabricGeneration(config_model,token="your_token") 
    model = fabric.setup(model)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config_model['lr'], weight_decay=config_model['weight_decay'], betas=(config_model['beta1'], config_model['beta2']), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, config_model, validation_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, config_model, validation_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    total_lengths = 0
    total_t0 = time.perf_counter()

    initial_iter = state["iter_num"]
    curr_iter = 0
    epochs = config_model["number_epochs"]
    train_iterator = trange(
        int(epochs), desc="Epoch", disable=False, mininterval=0)
    epoch = 0
    for _ in train_iterator:

        # This will print the info only for the first epoch
        if fabric.global_rank == 0:
            train_iterator.set_description(
                f"Running Epoch {epoch + 1} of {epochs}"
            )

        # This will print the progress bar only for the main process
        if fabric.global_rank == 0:
            batch_iterator = tqdm(train_dataloader, mininterval=0, colour="blue")
        else:
            batch_iterator = train_dataloader

        
        half_batches = int((len(batch_iterator) / 2) // config_model['gradient_accumulation_steps'])
        """ print("Evaluation batch" + str(half_batches)) """
        eval_step_interval = initial_iter + half_batches
        #eval_step_interval = 5

        # Iterate over each batch, getting the specific index of that batch
        for step, batch in enumerate(batch_iterator):
            # resume loader state. This is not elegant but it works. Should rewrite it in the future.
            if resume:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                else:
                    resume = False
                    curr_iter = -1
                    fabric.barrier()
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
            if state["iter_num"] >= max_iters:
                break

            # determine and set the learning rate for this iteration
            lr = get_lr(state["iter_num"], config_model["lr"]) if config_model["decay_lr"] else config_model["lr"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            is_accumulating = (state["iter_num"] + 1) % config_model["gradient_accumulation_steps"] != 0

            # with fabric.no_backward_sync(model, enabled=is_accumulating): #This is used in FSDP
            loss = model.training_step(batch, step)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0) #This is used in FSDP
            fabric.backward(loss / config_model["gradient_accumulation_steps"])

            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=config_model["grad_clip"])
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
            state["iter_num"] += 1
            # input_id: B L
            total_lengths += batch["input_ids"].size(1)
            t1 = time.perf_counter()
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    # print days as well
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                )

            monitor.on_train_batch_end(
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                state["step_count"],
                #flops_per_batch=estimated_flops,
                lengths=total_lengths,
                train_loss=loss.item()
            )


            """ if validation_dataloader is not None and state["step_count"] == eval_step_interval:   #  

                t0 = time.perf_counter()
                val_loss = validate(fabric, model, validation_dataloader)
                t1 = time.perf_counter() - t0
                monitor.eval_end(t1)
                fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
                fabric.log_dict({"metric/val_loss": val_loss.item(), }, state["step_count"])   # "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size
                fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), }, state["step_count"])   # "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size
                fabric.barrier() """


            if not is_accumulating and state["step_count"] % config_model['save_step_interval'] == 0:
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)

        ######## Validation ########
        t0 = time.perf_counter()
        val_loss = validate(fabric, model, validation_dataloader)
        t1 = time.perf_counter() - t0
        monitor.eval_end(t1)
        fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
        fabric.log_dict({"metric/val_loss": val_loss.item()}, state["step_count"])   # "total_tokens": model.config.block_size * ( state["iter_num"] + 1) * micro_batch_size * fabric.world_size
        fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), }, state["step_count"]) # "total_tokens": model.config.block_size * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size
        fabric.barrier()
        ######## Validation ########

        checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, state)


        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    batch_iterator = tqdm(val_dataloader, desc=f"Running Epoch 1 of 1", mininterval=0,
                               colour="green")

    for k, val_data in enumerate(batch_iterator):
        if k >= eval_iters:
            break
        outputs = model.validation_step(val_data, k)
        losses[k] = outputs.loss
        
    out = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    #torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
