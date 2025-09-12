#!/bin/bash

# Qwen3 Knowledge Distillation Script
# 단일 GPU 환경에서 실행

# 기본 설정
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 모델 및 데이터 설정
BASE_MODEL="Qwen/Qwen3-8B"
CACHE_DIR="./cache"
OUTPUT_DIR="./output/qwen3_distill_experiment"
TASK_TYPE="sequential"
DOMAIN_TYPE="music"

# 훈련 하이퍼파라미터
BATCH_SIZE=8
MICRO_BATCH_SIZE=1
NUM_EPOCHS=3
LEARNING_RATE=5e-6
MAX_STEPS=-1
SAVE_STEPS=50
EVAL_STEPS=50
WARMUP_STEPS=200

# LoRA 설정
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 지식 증류 설정
QWEN_DECODER_NUMS_TEACHER=28  # Qwen3-8B의 전체 레이어 수
QWEN_DECODER_NUMS_STUDENT=14  # 학생 모델은 절반 레이어
DISTILL_BLOCK=4
DISTILL_LAMBDA=1.0
DISTILL_TYPE="other"
DISTILL_TYPE_STANDARD="offline"  # offline 또는 online

# 체크포인트 설정 (필요시 수정)
TEACHER_CHECKPOINT=""
STUDENT_CHECKPOINT=""

# GPU 디바이스 설정
GPU_DEVICE=0

# Offline Distillation 실행
echo "Starting Qwen3 Knowledge Distillation Training..."
echo "Teacher layers: $QWEN_DECODER_NUMS_TEACHER"
echo "Student layers: $QWEN_DECODER_NUMS_STUDENT"
echo "Distillation type: $DISTILL_TYPE_STANDARD"

python distill_qwen3.py \
    --base_model $BASE_MODEL \
    --cache_dir $CACHE_DIR \
    --output_dir $OUTPUT_DIR \
    --task_type $TASK_TYPE \
    --domain_type $DOMAIN_TYPE \
    --gpu_device $GPU_DEVICE \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --qwen_decoder_nums_teacher $QWEN_DECODER_NUMS_TEACHER \
    --qwen_decoder_nums_student $QWEN_DECODER_NUMS_STUDENT \
    --distill_block $DISTILL_BLOCK \
    --distill_lambda $DISTILL_LAMBDA \
    --distill_type $DISTILL_TYPE \
    --distill_type_standard $DISTILL_TYPE_STANDARD \
    --train_eval_type "train" \
    --train_stargy "lora" \
    --lr_scheduler "linear" \
    --prompt_template_name "alpaca"

echo "Qwen3 Knowledge Distillation Training Completed!"

# 평가만 실행하려면 다음과 같이 수정:
# --train_eval_type "eval" \
# --student_resume_from_checkpoint "path/to/student/checkpoint"
