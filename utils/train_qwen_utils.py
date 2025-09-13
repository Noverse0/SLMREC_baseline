import transformers
import os
from typing import Any, Dict, List, Optional, Union
import json
import torch
import torch.nn.functional as F
import math
from transformers.trainer import *

class DistillationTrainingArguments(transformers.TrainingArguments):
    def __init__(self, *args, distill_lambda=0.001, qwen_decoder_nums_student=14, qwen_decoder_nums_teacher=28, distill_block=4, distill_type="other", distill_leave_layers=0, distill_type_standard="offline",
                is_cls_multiple=False,
                cls_multiple_lambda=1.0,
                kd_loss_type="cosine",
                is_cls_multiple_teacher=False,
                is_cls_multiple_student=False,
                cls_multiple_lambda_teacher=1.0,
                cls_multiple_lambda_student=1.0, 
                # Keep backward compatibility with llama parameter names
                llama_decoder_nums_student=None,
                llama_decoder_nums_teacher=None,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_lambda = distill_lambda
        
        # Handle backward compatibility - use qwen parameters if provided, otherwise fall back to llama
        if llama_decoder_nums_student is not None and qwen_decoder_nums_student == 14:
            self.qwen_decoder_nums_student = llama_decoder_nums_student
        else:
            self.qwen_decoder_nums_student = qwen_decoder_nums_student
            
        if llama_decoder_nums_teacher is not None and qwen_decoder_nums_teacher == 28:
            self.qwen_decoder_nums_teacher = llama_decoder_nums_teacher
        else:
            self.qwen_decoder_nums_teacher = qwen_decoder_nums_teacher
            
        # Keep llama parameter names for backward compatibility
        self.llama_decoder_nums_student = self.qwen_decoder_nums_student
        self.llama_decoder_nums_teacher = self.qwen_decoder_nums_teacher
        
        self.distill_block = distill_block
        self.distill_type = distill_type
        self.distill_leave_layers = distill_leave_layers
        self.distill_type_standard = distill_type_standard
        self.is_cls_multiple=is_cls_multiple
        self.cls_multiple_lambda=cls_multiple_lambda
        self.kd_loss_type=kd_loss_type
        self.is_cls_multiple_teacher=is_cls_multiple_teacher
        self.is_cls_multiple_student=is_cls_multiple_student
        self.cls_multiple_lambda_teacher=cls_multiple_lambda_teacher
        self.cls_multiple_lambda_student=cls_multiple_lambda_student

class SLMTrainer(transformers.Trainer):

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (optional):
                Start time for timing information (compatibility with parent class).
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        # save output
        with open(os.path.join(self.args.output_dir,"log.txt"), 'a') as file:
            # print("logger output:{}".format(output))
            json.dump(output, file)
            file.write('\n')

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

class DistillationTrainer(transformers.Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        if teacher_model is not None:
            # For Qwen3, check if the model has qwen_model attribute, otherwise use llama_model for compatibility
            if hasattr(self.model, 'qwen_model'):
                target_device = self.model.qwen_model.device
            elif hasattr(self.model, 'llama_model'):
                target_device = self.model.llama_model.device
            else:
                # Fallback to first available device
                target_device = next(self.model.parameters()).device
                
            self._move_model_to_device(self.teacher, target_device)
            self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

    def load_teacher_checkpoint(self, checkpoint_path, teacher_model):
        """
        Safe method to load teacher checkpoint avoiding _keys_to_ignore_on_save issue
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Teacher checkpoint path does not exist: {checkpoint_path}")
            return
            
        try:
            print(f"Loading teacher checkpoint from: {checkpoint_path}")
            
            # Method 1: Try standard trainer checkpoint loading
            try:
                self._load_from_checkpoint(checkpoint_path, model=teacher_model)
                print("Successfully loaded teacher checkpoint using trainer method")
                return
            except AttributeError as e:
                print(f"Trainer method failed with AttributeError: {e}")
                print("Trying alternative loading method...")
            
            # Method 2: Direct model weight loading
            model_files = [
                "pytorch_model.bin",
                "model.safetensors",
                "adapter_model.bin",  # For LoRA adapters
                "adapter_model.safetensors"
            ]
            
            checkpoint_loaded = False
            for model_file in model_files:
                model_path = os.path.join(checkpoint_path, model_file)
                if os.path.exists(model_path):
                    try:
                        if model_file.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(model_path)
                        else:
                            state_dict = torch.load(model_path, map_location='cpu')
                        
                        # Load state dict with non-strict matching
                        missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
                        
                        if missing_keys:
                            print(f"Missing keys when loading {model_file}: {missing_keys[:5]}...")  # Show first 5
                        if unexpected_keys:
                            print(f"Unexpected keys when loading {model_file}: {unexpected_keys[:5]}...")  # Show first 5
                            
                        print(f"Successfully loaded teacher checkpoint from: {model_file}")
                        checkpoint_loaded = True
                        break
                        
                    except Exception as e:
                        print(f"Failed to load {model_file}: {e}")
                        continue
            
            if not checkpoint_loaded:
                print(f"Could not find any loadable model files in: {checkpoint_path}")
                print(f"Available files: {os.listdir(checkpoint_path) if os.path.exists(checkpoint_path) else 'Directory not found'}")
                
        except Exception as e:
            print(f"Error loading teacher checkpoint: {e}")


class RecDistillationTrainer(DistillationTrainer,SLMTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # num_items_in_batch 매개변수는 kwargs에서 처리되므로 무시됨
        
        if self.args.distill_type_standard == "offline":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            try:
                if(torch.max(outputs_student['data_type']).item() ==0):
                    student_loss = outputs_student['loss']
                    # compute teacher output
                    with torch.no_grad():
                        outputs_teacher = self.teacher(**inputs)

                    # assert size
                    # assert outputs_student.logits.size() == outputs_teacher.logits.size()
                    loss_distill = 0
                    teacher_output_states = outputs_teacher['teacher_output_states']
                    student_hidden_states = outputs_student['student_output_states']
                    if student_hidden_states is not None:
                        if self.args.distill_type=="align": # block-wise alignment
                            for i in range(1,self.args.distill_block+1): # 1-4 block
                                # Use qwen decoder numbers for Qwen3 models
                                student_layer_idx = (self.args.qwen_decoder_nums_student//self.args.distill_block)*i
                                teacher_layer_idx = (self.args.qwen_decoder_nums_teacher//self.args.distill_block)*i
                                
                                cosine_sim = F.cosine_similarity(
                                    student_hidden_states[student_layer_idx][:,-1], 
                                    teacher_output_states[teacher_layer_idx][:,-1], 
                                    dim=1
                                )
                                l2_distance = torch.norm(
                                    student_hidden_states[student_layer_idx][:,-1] - teacher_output_states[teacher_layer_idx][:,-1], 
                                    dim=1, p=2
                                ).mean()
                                loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda + l2_distance * 0.1
                        loss_distill_dict = {"loss_distill":loss_distill.item()}
                        self.log(loss_distill_dict)
                    else:
                        loss_distill = 0
                    if self.args.is_cls_multiple:
                        if outputs_student['loss_cls_multiple'] is not None:
                            loss_multiple = outputs_student['loss_cls_multiple'] * self.args.cls_multiple_lambda
                            student_loss = student_loss + loss_multiple
                            loss_multiple_dict = {"loss_multiple":loss_multiple.item()}
                            self.log(loss_multiple_dict)
                    loss = student_loss + loss_distill
            except:
                loss = outputs_student['loss']
        elif self.args.distill_type_standard=="online":
            # compute student output
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs_student = model(**inputs)
            try:
                if torch.max(outputs_student['data_type']).item() ==0:
                    student_loss = outputs_student['loss']
                    teacher_output_states = outputs_student['teacher_output_states']
                    student_hidden_states = outputs_student['student_output_states']

                    loss_distill = 0
                    if self.args.kd_loss_type == "cosine":
                        if teacher_output_states is not None and student_hidden_states is not None:
                            if self.args.distill_type=="align": # block-wise alignment
                                for i in range(1,self.args.distill_block+1): # 1-4 block
                                    # Use qwen decoder numbers for Qwen3 models
                                    student_layer_idx = (self.args.qwen_decoder_nums_student//self.args.distill_block)*i
                                    teacher_layer_idx = (self.args.qwen_decoder_nums_teacher//self.args.distill_block)*i
                                    
                                    cosine_sim = F.cosine_similarity(
                                        student_hidden_states[student_layer_idx][:,-1], 
                                        teacher_output_states[teacher_layer_idx][:,-1], 
                                        dim=1
                                    )
                                    l2_distance = torch.norm(
                                        student_hidden_states[student_layer_idx][:,-1] - teacher_output_states[teacher_layer_idx][:,-1], 
                                        dim=1, p=2
                                    ).mean()
                                    loss_distill = loss_distill + (1 - cosine_sim.mean()) * self.args.distill_lambda + l2_distance * 0.1

                            loss_distill_dict = {"loss_distill":loss_distill.item()}
                            self.log(loss_distill_dict)
                        else:
                            loss_distill = 0
                    elif self.args.kd_loss_type == "logit":
                        for i in range(0,self.args.distill_block):
                            if outputs_student['logits_teacher'] is not None and outputs_student['logits_student'] is not None: 
                                logits_teacher_tmp, logits_student_tmp = outputs_student['logits_teacher'][i], outputs_student['logits_student'][i] 
                                teacher_probs = F.softmax(logits_teacher_tmp, dim=1)
                                student_probs_with_log = F.log_softmax(logits_student_tmp, dim=1)
                                kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
                                loss_distill = loss_distill + kl_loss(student_probs_with_log, teacher_probs.detach()) * self.args.distill_lambda
                                loss_distill_dict = {"loss_distill_kl":loss_distill.item()}
                                self.log(loss_distill_dict)

                    if self.args.is_cls_multiple_teacher:
                        if outputs_student['loss_cls_multiple_teacher'] is not None:
                            loss_multiple_teacher = outputs_student['loss_cls_multiple_teacher'] * self.args.cls_multiple_lambda_teacher
                            student_loss = student_loss + loss_multiple_teacher
                            loss_multiple_teacher_dict = {"loss_multiple_teacher":loss_multiple_teacher.item()}
                            self.log(loss_multiple_teacher_dict)

                    if self.args.is_cls_multiple_student:
                        if outputs_student['loss_cls_multiple_student'] is not None:
                            loss_multiple_student = outputs_student['loss_cls_multiple_student'] * self.args.cls_multiple_lambda_student
                            student_loss = student_loss + loss_multiple_student
                            loss_multiple_student_dict = {"loss_multiple_student":loss_multiple_student.item()}
                            self.log(loss_multiple_student_dict)
                    
                    loss = student_loss + loss_distill
            except:
                loss = outputs_student['loss']
        return (loss, outputs_student) if return_outputs else loss