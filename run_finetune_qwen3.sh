#!/bin/bash
# Get the output directory and GPU device from command line arguments
OUTPUT_DIR=$1
GPU_DEVICE=${2:-1}  # Default to GPU 1 if not specified

# Check if the output directory parameter is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required."
    echo "Usage: $0 <output_directory> [gpu_device_number]"
    echo "Example: $0 output_qwen3_test 1"
    exit 1
fi

export BITSANDBYTES_NOWELCOME=1

# --base_model "Qwen/Qwen3-14B" \
# --lora_target_modules '[gate_proj, down_proj, up_proj]' \
# Single GPU training with Qwen3-14B
echo "Starting Qwen3-14B-based SLMRec training on GPU $GPU_DEVICE..."
echo "Note: Qwen3-14B requires significant GPU memory (28GB+)"

# Set environment variable to use only specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

python finetune_qwen3.py \
    --base_model "Qwen/Qwen3-8B" \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --gpu_device $GPU_DEVICE \
    --batch_size 128 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 0.001 \
    --cutoff_len 4096 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["gate_proj", "down_proj", "up_proj"]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 100 \
    --lr_scheduler cosine \
    --qwen_decoder_nums 8 \
    --save_steps 100 \
    --eval_steps 100 \
    --domain_type toys

echo "Qwen3-14B training completed!" 