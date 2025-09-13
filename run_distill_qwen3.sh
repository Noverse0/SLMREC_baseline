#!/bin/bash

# Qwen3 Knowledge Distillation Script
# 단일 GPU 환경에서 실행

# 기본 설정
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# 모델 및 데이터 설정
BASE_MODEL="Qwen/Qwen3-8B"  # 기본 Qwen3 모델 (student 초기화용)
CACHE_DIR="./cache"
OUTPUT_DIR="./output/qwen3_student_distilled"  # student 모델 저장 경로
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
QWEN_DECODER_NUMS_TEACHER=8  
QWEN_DECODER_NUMS_STUDENT=4  
DISTILL_BLOCK=4
DISTILL_LAMBDA=1.0
DISTILL_TYPE="other"
DISTILL_TYPE_STANDARD="offline"

# 체크포인트 설정 - 로컬 파인튜닝된 모델을 teacher로 사용
TEACHER_CHECKPOINT="/home/daeyoung_roh/SLMRec/slmrec/music/teacher_model_qwen3/checkpoint-1380"
STUDENT_CHECKPOINT=""  # 새로 훈련할 것이므로 비워둠

# GPU 디바이스 설정
GPU_DEVICE=1

# Offline Distillation 실행
echo "Starting Qwen3 Knowledge Distillation Training..."
echo "Teacher checkpoint: $TEACHER_CHECKPOINT"
echo "Student output dir: $OUTPUT_DIR"
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
    --teacher_resume_from_checkpoint $TEACHER_CHECKPOINT \
    --train_eval_type "train" \
    --train_stargy "lora" \
    --lr_scheduler "linear" \
    --prompt_template_name "alpaca"

echo "Qwen3 Knowledge Distillation Training Completed!"
echo "Student model saved to: $OUTPUT_DIR"