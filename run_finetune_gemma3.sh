#!/bin/bash
# Get the output directory from command line argument
OUTPUT_DIR=$1

# Check if the output directory parameter is provided
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory is required."
    echo "Usage: $0 <output_directory>"
    exit 1
fi

export BITSANDBYTES_NOWELCOME=1

# Single GPU training with Gemma3
echo "Starting Gemma3-based SLMRec training on single GPU..."
echo "Note: Gemma3 9B requires significant GPU memory (20GB+)"
python finetune_gemma3.py \
    --base_model "google/gemma-2-9b" \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 0.0005 \
    --cutoff_len 4096 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --warmup_steps 100 \
    --lr_scheduler cosine \
    --gemma_decoder_nums 42 \
    --save_steps 100 \
    --eval_steps 100 \
    --domain_type music

echo "Gemma3 training completed!" 