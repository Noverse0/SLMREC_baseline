import os
import sys
from typing import List
import argparse
import json

import fire
import torch
import pickle
import numpy as np
import transformers
from transformers import AutoTokenizer, GemmaModel
from utils.prompter import Prompter
from model_gemma3 import LLM4RecGemma3
from utils.data_utils import *
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel, compute_metrics
from utils.train_utils import SLMTrainer

def train(
    # model/data params
    base_model: str = "google/gemma-2-9b", 
    data_path: str = "",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "",
    train_stargy: str = "lora",
    # training hyperparams
    batch_size: int = 64,  # Reduced for Gemma3 9B
    micro_batch_size: int = 4,  # Smaller for larger model
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    interval_nums: int = 0,
    drop_type: str = "trune",
    lr_scheduler: str = "cosine",
    max_steps: int = -1,
    warmup_steps: int = 100, 
    save_steps: int = 100,
    eval_steps: int = 100,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Gemma3 target modules for LoRA
    lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",
    gemma_decoder_nums: int = 42,  # Gemma3 9B has 42 layers
    domain_type: str = "cloths",
):
    print(
        f"Params using prompt template {prompt_template_name}:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"cache_dir: {cache_dir}\n"
        f"output_dir: {output_dir}\n"
        f"task_type: {task_type}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lr_scheduler: {lr_scheduler}\n"
        f"warmup_steps: {warmup_steps}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        f"gemma_decoder_nums: {gemma_decoder_nums}\n"
        f"domain_type: {domain_type}\n"
    )

    # Single GPU setup - no DDP
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"

    prompter = Prompter(prompt_template_name)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Load item embeddings
    # choose from cloths and movies
    # item_embed = pickle.load(open('./sasrec_cloths/sasrec_item.pkl', 'rb'))['item_embedding']
    item_embed = pickle.load(open('./output/music.pkl', 'rb'))['item_embedding']
            
    # Initialize Gemma3-based model with float16 precision
    model = LLM4RecGemma3(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=128,
        output_dim=0,
        interval_nums=interval_nums,
        drop_type=drop_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        train_stargy=train_stargy,
        user_embeds=None,
        input_embeds=item_embed,
        seq_len=30,
        gemma_decoder_nums=gemma_decoder_nums,
    )

    # Single GPU setup - no model parallelism needed
    if torch.cuda.device_count() > 1:
        print(f"Warning: Multiple GPUs detected ({torch.cuda.device_count()}), but using single GPU mode")
        print("Set CUDA_VISIBLE_DEVICES to use specific GPU if needed")

    # Load datasets
    datasetTrain = LLMDataset(item_size=999, max_seq_length=30, data_type='train', csv_path="./dataset/sequential/{}.csv".format(domain_type))
    datasetVal = LLMDataset(item_size=999, max_seq_length=30, data_type='valid', csv_path="./dataset/sequential/{}.csv".format(domain_type))
    datasetTest = LLMDataset(item_size=999, max_seq_length=30, data_type='test', csv_path="./dataset/sequential/{}.csv".format(domain_type))
    data_collator = SequentialCollator()

    # Set save and evaluation strategies
    if save_steps < 0:
        save_strategy = "epoch"
    else:
        save_strategy = "steps"
    if eval_steps < 0:
        evaluation_strategy = "epoch"
    else:
        evaluation_strategy = "steps"

    # Initialize trainer
    trainer = SLMTrainer(
        model=model,
        train_dataset=datasetTrain,
        eval_dataset=datasetVal,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            include_inputs_for_metrics=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            dataloader_num_workers=2,  # Further reduced for 9B model
            per_device_eval_batch_size=128,  # Reduced for larger model
            remove_unused_columns=False,
            max_steps=max_steps,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            metric_for_best_model="mrr",
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            lr_scheduler_type=lr_scheduler,
            logging_dir=output_dir,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            group_by_length=group_by_length,
            report_to="tensorboard",
            run_name=wandb_run_name if wandb_run_name else None,
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting Gemma3 training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Get best checkpoint path
    best_checkpoint_path = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint_path}")

    # Reload model from best checkpoint for evaluation
    print("Reloading best Gemma3 model for evaluation...")
    model = LLM4RecGemma3(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=128,
        output_dim=0,
        interval_nums=interval_nums,
        drop_type=drop_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        train_stargy=train_stargy,
        user_embeds=None,
        input_embeds=item_embed,
        seq_len=30,
        gemma_decoder_nums=gemma_decoder_nums,
    )

    # Create new trainer for evaluation
    trainer = SLMTrainer(
        model=model,
        train_dataset=datasetTrain,
        eval_dataset=datasetVal,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            include_inputs_for_metrics=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            dataloader_num_workers=2,
            per_device_eval_batch_size=128,
            remove_unused_columns=False,
            max_steps=max_steps,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            metric_for_best_model="mrr",
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            lr_scheduler_type=lr_scheduler,
            logging_dir=output_dir,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=False,
            group_by_length=group_by_length,
            report_to="tensorboard",
            run_name=wandb_run_name if wandb_run_name else None,
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Load best checkpoint and evaluate on test set
    trainer._load_from_checkpoint(best_checkpoint_path)
    print("Running evaluation on test set...")
    pred_out = trainer.predict(test_dataset=datasetTest)
    
    # Save evaluation results
    output_data = {}
    if pred_out.metrics is not None:
        for metric_name, metric_value in pred_out.metrics.items():
            print(f"{metric_name}: {metric_value}")
            output_data[metric_name] = metric_value

    # Write the output data to a file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "log.txt"), 'a') as file:
        json.dump(output_data, file, indent=2)
        file.write('\n')

    print(f"Gemma3 training completed! Results saved to {output_dir}/log.txt")

if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train) 