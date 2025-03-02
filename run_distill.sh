#!/bin/bash

# Get the parameters from command line arguments
TEACHER_CHECKPOINT=$1
OUTPUT_DIR=$2

# Check if both parameters are provided
if [ -z "$TEACHER_CHECKPOINT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Both teacher checkpoint path and output directory are required."
  echo "Usage: $0 <teacher_checkpoint_path> <output_directory>"
  exit 1
fi

# Run the command with the provided parameters
NCCL_P2P_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=8 --master_port=1234 distill.py \
    --task_type sequential \
    --cache_dir cace_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 256 \
    --micro_batch_size 32 \
    --num_epochs 5 \
    --learning_rate 0.001 \
    --cutoff_len 4096 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 50 \
    --lr_scheduler 'cosine' \
    --llama_decoder_nums_teacher 8 \
    --llama_decoder_nums_student 4 \
    --teacher_resume_from_checkpoint "$TEACHER_CHECKPOINT" \
    --distill_lambda 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --domain_type "cloths"