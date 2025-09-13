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
from transformers import AutoTokenizer, AutoModel
from utils.prompter import Prompter
from model_qwen3 import LLM4RecQwen3, LLM4RecQwen3Teacher, LLM4RecQwen3Student, LLM4RecQwen3Distill
from utils.data_utils import *
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel, compute_metrics
from utils.train_qwen_utils import RecDistillationTrainer, DistillationTrainingArguments

def train(
    # model/data params
    base_model: str = "Qwen/Qwen3-8B", 
    data_path: str = "",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "",
    train_stargy: str = "lora",
    gpu_device: int = 0,
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    interval_nums: int = 0,
    drop_type: str = "trune",
    lr_scheduler: str = "linear",
    warmup_steps: int = 200,
    max_steps: int = -1,
    save_steps: int = 50,
    eval_steps: int = 50,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Qwen3 target modules for LoRA
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    teacher_resume_from_checkpoint: str = None,
    student_resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    domain_type: str = "music",
    qwen_decoder_nums_teacher: int = 28,  # Qwen3-8B has 28 layers
    qwen_decoder_nums_student: int = 14,  # Student model with half layers
    distill_block: int = 4,
    distill_lambda: float = 1.0,
    distill_type: str = "other",
    distill_type_standard: str = "offline",
    is_cls_multiple: bool = False,
    train_eval_type: str = "train",
    cls_multiple_lambda: float = 1.0,
    kd_loss_type: str = "cosine",
    is_cls_multiple_teacher: bool = False,
    is_cls_multiple_student: bool = False,
    cls_multiple_lambda_teacher: float = 1.0,
    cls_multiple_lambda_student: float = 1.0,
):
    # Set single GPU usage - 단일 GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Using GPU device: {gpu_device}")
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

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
        f"teacher_resume_from_checkpoint: {teacher_resume_from_checkpoint}\n"
        f"student_resume_from_checkpoint: {student_resume_from_checkpoint}\n"
        f"qwen_decoder_nums_teacher: {qwen_decoder_nums_teacher}\n"
        f"qwen_decoder_nums_student: {qwen_decoder_nums_student}\n"
        f"distill_lambda: {distill_lambda}\n"
        f"distill_block: {distill_block}\n"
        f"domain_type: {domain_type}\n"
        f"distill_type: {distill_type}\n"
        f"distill_type_standard: {distill_type_standard}\n"
        f"is_cls_multiple: {is_cls_multiple}\n"
        f"cls_multiple_lambda: {cls_multiple_lambda}\n"
        f"kd_loss_type: {kd_loss_type}\n"
        f"is_cls_multiple_teacher: {is_cls_multiple_teacher}\n"
        f"is_cls_multiple_student: {is_cls_multiple_student}\n"
        f"cls_multiple_lambda_teacher: {cls_multiple_lambda_teacher}\n"
        f"cls_multiple_lambda_student: {cls_multiple_lambda_student}\n"
    )

    # Single GPU setup - no DDP
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = {"": 0}  # Force all layers to GPU 0

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
    # item_embed = pickle.load(open('./sasrec_{}/sasrec_item.pkl'.format(domain_type), 'rb'))['item_embedding']
    item_embed = pickle.load(open('./output/{}.pkl'.format(domain_type), 'rb'))['item_embedding']

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
        eval_strategy = "epoch"
    else:
        eval_strategy = "steps"

    if distill_type_standard == "offline":
        # Offline distillation: separate teacher and student models
        model_teacher = LLM4RecQwen3Teacher(
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
            qwen_decoder_nums=qwen_decoder_nums_teacher,
        )
        
        model = LLM4RecQwen3Student(
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
            qwen_decoder_nums=qwen_decoder_nums_student,
            distill_block=distill_block,
            is_cls_multiple=is_cls_multiple,
        )
        
        trainer = RecDistillationTrainer(
            teacher_model=model_teacher,
            model=model,
            train_dataset=datasetTrain,
            eval_dataset=datasetVal,
            args=DistillationTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                include_inputs_for_metrics=True,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                dataloader_num_workers=0,  # Single GPU optimization
                per_device_eval_batch_size=128,
                remove_unused_columns=False,
                fp16=False,
                bf16=True,  # Use bf16 for Qwen3
                logging_steps=1,
                optim="adamw_torch",
                max_grad_norm=1.0,
                weight_decay=0.01,
                metric_for_best_model="mrr",
                eval_strategy=eval_strategy,
                save_strategy=save_strategy,
                max_steps=max_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                lr_scheduler_type=lr_scheduler,
                logging_dir=output_dir,
                output_dir=output_dir,
                save_total_limit=2,
                load_best_model_at_end=False,
                group_by_length=group_by_length,
                report_to="tensorboard",
                run_name=None,
                distill_lambda=distill_lambda,
                llama_decoder_nums_student=qwen_decoder_nums_student,
                llama_decoder_nums_teacher=qwen_decoder_nums_teacher,
                distill_block=distill_block,
                distill_type=distill_type,
                distill_type_standard=distill_type_standard,
                is_cls_multiple=is_cls_multiple,
                cls_multiple_lambda=cls_multiple_lambda,
                kd_loss_type=kd_loss_type,
            ),
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Load teacher checkpoint if provided
        if teacher_resume_from_checkpoint:
            trainer._load_from_checkpoint(teacher_resume_from_checkpoint, model=model_teacher)
        model_teacher.eval()
        
    elif distill_type_standard == "online":
        # Online distillation: combined teacher-student model
        model = LLM4RecQwen3Distill(
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
            item_embed=item_embed,
            seq_len=30,
            qwen_decoder_nums=qwen_decoder_nums_teacher,
            qwen_decoder_nums_teacher=qwen_decoder_nums_teacher,
            qwen_decoder_nums_student=qwen_decoder_nums_student,
            distill_lambda=distill_lambda,
            distill_block=distill_block,
            is_cls_multiple_teacher=is_cls_multiple_teacher,
            is_cls_multiple_student=is_cls_multiple_student,
        )
        
        trainer = RecDistillationTrainer(
            teacher_model=None,  # Covered in student model
            model=model,
            train_dataset=datasetTrain,
            eval_dataset=datasetVal,
            args=DistillationTrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                include_inputs_for_metrics=True,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                dataloader_num_workers=0,  # Single GPU optimization
                per_device_eval_batch_size=128,
                remove_unused_columns=False,
                fp16=False,
                bf16=True,  # Use bf16 for Qwen3
                logging_steps=1,
                optim="adamw_torch",
                max_grad_norm=1.0,
                weight_decay=0.01,
                metric_for_best_model="mrr",
                eval_strategy=eval_strategy,
                save_strategy=save_strategy,
                max_steps=max_steps,
                eval_steps=eval_steps,
                save_steps=save_steps,
                lr_scheduler_type=lr_scheduler,
                logging_dir=output_dir,
                output_dir=output_dir,
                save_total_limit=2,
                load_best_model_at_end=False,
                group_by_length=group_by_length,
                report_to="tensorboard",
                run_name=None,
                distill_lambda=distill_lambda,
                llama_decoder_nums_student=qwen_decoder_nums_student,
                llama_decoder_nums_teacher=qwen_decoder_nums_teacher,
                distill_block=distill_block,
                distill_type=distill_type,
                distill_type_standard=distill_type_standard,
                is_cls_multiple=is_cls_multiple,
                is_cls_multiple_teacher=is_cls_multiple_teacher,
                is_cls_multiple_student=is_cls_multiple_student,
                cls_multiple_lambda=cls_multiple_lambda,
                kd_loss_type=kd_loss_type,
                cls_multiple_lambda_teacher=cls_multiple_lambda_teacher,
                cls_multiple_lambda_student=cls_multiple_lambda_student,
            ),
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    if train_eval_type == "train":
        print("Starting Qwen3 distillation training...")
        trainer.train()
        best_checkpoint_path = trainer.state.best_model_checkpoint
        print(f"Best checkpoint: {best_checkpoint_path}")
    elif train_eval_type == "eval":
        if student_resume_from_checkpoint:
            trainer._load_from_checkpoint(student_resume_from_checkpoint)


    if train_eval_type == "train":
        # Reload the model and path for evaluation
        if distill_type_standard == "offline":
            model_teacher = LLM4RecQwen3Teacher(
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
                qwen_decoder_nums=qwen_decoder_nums_teacher,
            )
            
            model = LLM4RecQwen3Student(
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
                qwen_decoder_nums=qwen_decoder_nums_student,
                distill_block=distill_block,
                is_cls_multiple=is_cls_multiple,
            )
            
            trainer = RecDistillationTrainer(
                teacher_model=model_teacher,
                model=model,
                train_dataset=datasetTrain,
                eval_dataset=datasetVal,
                args=DistillationTrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    include_inputs_for_metrics=True,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_steps,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    dataloader_num_workers=0,
                    per_device_eval_batch_size=128,
                    remove_unused_columns=False,
                    fp16=False,
                    bf16=True,
                    logging_steps=1,
                    optim="adamw_torch",
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    metric_for_best_model="mrr",
                    eval_strategy=eval_strategy,
                    save_strategy=save_strategy,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    max_steps=max_steps,
                    lr_scheduler_type=lr_scheduler,
                    logging_dir=output_dir,
                    output_dir=output_dir,
                    save_total_limit=2,
                    load_best_model_at_end=False,
                    group_by_length=group_by_length,
                    report_to="tensorboard",
                    run_name=None,
                    distill_lambda=distill_lambda,
                    llama_decoder_nums_student=qwen_decoder_nums_student,
                    llama_decoder_nums_teacher=qwen_decoder_nums_teacher,
                    distill_block=distill_block,
                    distill_type=distill_type,
                    distill_type_standard=distill_type_standard,
                    is_cls_multiple=is_cls_multiple,
                    cls_multiple_lambda=cls_multiple_lambda,
                    kd_loss_type=kd_loss_type,
                ),
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            if teacher_resume_from_checkpoint:
                trainer._load_from_checkpoint(teacher_resume_from_checkpoint, model=model_teacher)
            model_teacher.eval()
            
        elif distill_type_standard == "online":
            model = LLM4RecQwen3Distill(
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
                item_embed=item_embed,
                seq_len=30,
                qwen_decoder_nums=qwen_decoder_nums_teacher,
                qwen_decoder_nums_teacher=qwen_decoder_nums_teacher,
                qwen_decoder_nums_student=qwen_decoder_nums_student,
                distill_lambda=distill_lambda,
                distill_block=distill_block,
                is_cls_multiple_teacher=is_cls_multiple_teacher,
                is_cls_multiple_student=is_cls_multiple_student,
            )
            
            trainer = RecDistillationTrainer(
                teacher_model=None,
                model=model,
                train_dataset=datasetTrain,
                eval_dataset=datasetVal,
                args=DistillationTrainingArguments(
                    per_device_train_batch_size=micro_batch_size,
                    include_inputs_for_metrics=True,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=warmup_steps,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    dataloader_num_workers=0,
                    per_device_eval_batch_size=128,
                    remove_unused_columns=False,
                    fp16=False,
                    bf16=True,
                    logging_steps=1,
                    optim="adamw_torch",
                    max_grad_norm=1.0,
                    weight_decay=0.01,
                    metric_for_best_model="mrr",
                    eval_strategy=eval_strategy,
                    save_strategy=save_strategy,
                    eval_steps=eval_steps,
                    save_steps=save_steps,
                    max_steps=max_steps,
                    lr_scheduler_type=lr_scheduler,
                    logging_dir=output_dir,
                    output_dir=output_dir,
                    save_total_limit=2,
                    load_best_model_at_end=False,
                    group_by_length=group_by_length,
                    report_to="tensorboard",
                    run_name=None,
                    distill_lambda=distill_lambda,
                    llama_decoder_nums_student=qwen_decoder_nums_student,
                    llama_decoder_nums_teacher=qwen_decoder_nums_teacher,
                    distill_block=distill_block,
                    distill_type=distill_type,
                    distill_type_standard=distill_type_standard,
                    is_cls_multiple=is_cls_multiple,
                    is_cls_multiple_teacher=is_cls_multiple_teacher,
                    is_cls_multiple_student=is_cls_multiple_student,
                    cls_multiple_lambda=cls_multiple_lambda,
                    kd_loss_type=kd_loss_type,
                    cls_multiple_lambda_teacher=cls_multiple_lambda_teacher,
                    cls_multiple_lambda_student=cls_multiple_lambda_student,
                ),
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        
        trainer._load_from_checkpoint(best_checkpoint_path)

    # Run evaluation on test set
    print("Running evaluation on test set...")
    pred_out = trainer.predict(test_dataset=datasetTest)
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

    print(f"Qwen3 distillation training completed! Results saved to {output_dir}/log.txt")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(train)
