# SLMRec with Gemma3 Support

이 버전은 기존 LLaMA 기반의 SLMRec을 Google의 Gemma3 모델로 변경한 단일 GPU 지원 버전입니다.

## 주요 변경사항

### 1. 모델 아키텍처
- **기존**: LLaMA-2-7B 기반
- **변경**: Gemma-2-9B 기반 (`google/gemma-2-9b`)

### 2. GPU 지원
- **기존**: 멀티 GPU (DDP) 지원
- **변경**: 단일 GPU 최적화 (고사양 GPU 권장)

### 3. 새로운 파일들
- `model_gemma3.py`: Gemma3 기반 모델 클래스들
- `finetune_gemma3.py`: 단일 GPU용 fine-tuning 스크립트
- `run_finetune_gemma3.sh`: 실행 스크립트
- `requirements_gemma3.txt`: Gemma3 관련 의존성 포함

## 시스템 요구사항

### ⚠️ 중요: GPU 메모리 요구사항
- **최소**: 20GB GPU 메모리 (RTX 4090, A100 등)
- **권장**: 24GB+ GPU 메모리 (A100, H100 등)
- **Gemma3 9B**: 기본적으로 더 많은 메모리가 필요합니다

### 지원 GPU
- ✅ NVIDIA A100 (40GB/80GB)
- ✅ NVIDIA H100
- ✅ RTX 4090 (24GB) - 메모리 최적화 필요
- ❌ RTX 3080/3090 (10GB/24GB) - 메모리 부족 가능성

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python3 -m venv gemma3_env
source gemma3_env/bin/activate  # Linux/Mac
```

2. 의존성 설치:
```bash
pip install -r requirements_gemma3.txt
```

## 사용법

### 1. 데이터 준비
기존과 동일하게 데이터를 준비합니다:
```bash
python train_sr_trad.py
python extract_emb.py
```

### 2. Teacher 모델 훈련 (Gemma3 기반)
```bash
chmod +x run_finetune_gemma3.sh
bash run_finetune_gemma3.sh ./output/gemma3_teacher_model/
```

### 3. 직접 실행
```bash
python finetune_gemma3.py \
    --base_model "google/gemma-2-9b" \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir ./output/gemma3_teacher/ \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 0.0005 \
    --gemma_decoder_nums 42 \
    --domain_type music
```

## 주요 매개변수

### Gemma3 전용 매개변수
- `--base_model`: `"google/gemma-2-9b"` (기본값)
- `--gemma_decoder_nums`: Gemma3 디코더 레이어 수 (기본값: 42)
- `--lora_target_modules`: Gemma3에 맞는 LoRA 타겟 모듈들

### 메모리 최적화 설정
- `--batch_size`: 32 (9B 모델용으로 감소)
- `--micro_batch_size`: 4 (더 작은 배치 크기)
- `--per_device_eval_batch_size`: 128 (자동 설정)
- `--dataloader_num_workers`: 2 (자동 설정)

## 모델 클래스 구조

### 1. LLM4RecGemma3
- 기본 Gemma3 기반 추천 모델
- LLaMA 대신 GemmaModel 사용
- AutoTokenizer로 토크나이저 초기화

### 2. LLM4RecGemma3Teacher
- Teacher 모델용 클래스
- Hidden states 출력 지원

### 3. LLM4RecGemma3Student
- Student 모델용 클래스 (Knowledge Distillation용)
- Multiple supervision 지원

## 성능 최적화

### 메모리 사용량
- float16 정밀도 사용
- LoRA를 통한 parameter-efficient fine-tuning
- 9B 모델에 최적화된 배치 크기

### 속도 최적화
- 불필요한 DDP 코드 제거
- 단일 GPU에 맞는 데이터로더 설정
- 효율적인 gradient accumulation

## 메모리 부족 해결법

### 1. 배치 크기 조정
```bash
# 메모리가 부족한 경우
--batch_size 16 --micro_batch_size 2
```

### 2. LoRA 파라미터 조정
```bash
# LoRA rank 감소
--lora_r 8 --lora_alpha 8
```

### 3. Gradient Checkpointing 활용
모델 초기화 시 자동으로 활성화됩니다.

### 4. 특정 GPU 사용
```bash
export CUDA_VISIBLE_DEVICES=0
bash run_finetune_gemma3.sh ./output/gemma3_teacher/
```

## 성능 비교

| 모델 | 파라미터 수 | GPU 메모리 | 훈련 시간 | 추천 성능 |
|------|-------------|------------|-----------|-----------|
| LLaMA-2-7B | 7B | 14GB | 기준 | 기준 |
| Qwen2.5-7B | 7B | 14GB | 비슷 | 약간 향상 |
| **Gemma-2-9B** | **9B** | **20GB+** | **느림** | **높음** |

## 주의사항

1. **GPU 메모리**: Gemma3 9B는 최소 20GB의 GPU 메모리가 필요합니다.
2. **훈련 시간**: 더 큰 모델로 인해 훈련 시간이 길어집니다.
3. **토크나이저**: Gemma3는 자체 토크나이저를 사용합니다.
4. **LoRA 모듈**: Gemma3의 아키텍처에 맞게 LoRA target modules가 조정되었습니다.

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기를 더욱 감소
--batch_size 8 --micro_batch_size 1

# 또는 더 작은 모델 사용 고려
# Gemma-2-2B 버전 고려 (향후 지원 예정)
```

### 모델 다운로드 문제
```bash
# Hugging Face 토큰 설정 (Gemma는 제한된 모델)
export HF_TOKEN="your_token_here"

# 캐시 디렉토리 설정
export HF_HOME=/path/to/large/cache
```

### 토크나이저 관련 오류
```bash
# 최신 transformers 설치
pip install transformers>=4.45.2
```

## 예상 성능

- **메모리 효율성**: LoRA + float16 + gradient checkpointing
- **추천 성능**: LLaMA 대비 5-10% 향상 예상
- **훈련 시간**: 9B 모델로 인해 약 30-40% 증가

## 모델 접근 권한

Gemma 모델은 제한된 접근 권한이 필요할 수 있습니다:

1. Hugging Face 계정 생성
2. Gemma 모델 라이선스 동의
3. HF 토큰 생성 및 설정

```bash
# 토큰 설정
export HF_TOKEN="your_huggingface_token"
```

## 향후 계획

- [ ] Gemma-2-2B 소형 모델 지원
- [ ] Knowledge Distillation 지원 (Gemma3 Teacher → 소형 Student)
- [ ] 더 효율적인 메모리 최적화
- [ ] Multi-GPU 지원 (선택사항)

## 라이선스

Gemma 모델은 Google의 라이선스를 따릅니다. 상업적 사용 시 라이선스 조건을 확인하세요. 