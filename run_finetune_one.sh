#!/bin/bash
# Get the output directory and GPU device from command line arguments
OUTPUT_DIR=$1
GPU_DEVICE=${2:-0}  # Default to GPU 0 if not specified

# Check if the output directory parameter is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required."
    echo "Usage: $0 <output_directory> [gpu_device_number]"
    echo "Example: $0 output_test 1"
    exit 1
fi

export BITSANDBYTES_NOWELCOME=1

# Run the command with the provided output directory (Single GPU)
echo "Starting training on GPU $GPU_DEVICE..."
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python finetune.py \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --micro_batch_size 32 \
    --num_epochs 3 \
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
    --warmup_steps 100 \
    --lr_scheduler 'cosine' \
    --llama_decoder_nums 8 \
    --domain_type 'music'