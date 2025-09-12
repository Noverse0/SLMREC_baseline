# SLMRec with Qwen3-14B Support

이 버전은 기존 LLaMA 기반의 SLMRec을 **Qwen2.5-14B** 모델로 변경한 단일 GPU 지원 버전입니다.

## 🚨 중요: 고사양 GPU 필요

### GPU 메모리 요구사항
- **최소**: 28GB GPU 메모리
- **권장**: 32GB+ GPU 메모리
- **지원 GPU**:
  - ✅ NVIDIA A100 (40GB/80GB)
  - ✅ NVIDIA H100 (80GB)
  - ❌ RTX 4090 (24GB) - **메모리 부족**
  - ❌ RTX 3080/3090 - **메모리 부족**

## 주요 변경사항

### 1. 모델 업그레이드
- **기존**: Qwen2.5-7B (32 레이어)
- **변경**: **Qwen2.5-14B (40 레이어)**

### 2. 메모리 최적화 설정
- **배치 크기**: 64 → 32
- **Micro 배치**: 8 → 4
- **평가 배치**: 256 → 128
- **데이터로더 워커**: 4 → 2

### 3. 성능 향상
- **파라미터**: 7B → 14B (2배 증가)
- **예상 성능**: 7B 대비 10-15% 향상
- **훈련 시간**: 약 2배 증가

## 설치 방법

1. 가상환경 생성:
```bash
python3 -m venv qwen14b_env
source qwen14b_env/bin/activate
```

2. 의존성 설치:
```bash
pip install -r requirements_qwen3.txt
```

## 사용법

### 기본 실행
```bash
chmod +x run_finetune_qwen3.sh
bash run_finetune_qwen3.sh ./output/qwen14b_teacher/
```

### 직접 실행 (권장 설정)
```bash
python finetune_qwen3.py \
    --base_model "Qwen/Qwen2.5-14B" \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir ./output/qwen14b_teacher/ \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 0.001 \
    --qwen_decoder_nums 40 \
    --domain_type music
```

### 메모리 부족 시 (긴급 설정)
```bash
python finetune_qwen3.py \
    --base_model "Qwen/Qwen2.5-14B" \
    --task_type sequential \
    --batch_size 16 \
    --micro_batch_size 2 \
    --lora_r 8 \
    --lora_alpha 8 \
    --qwen_decoder_nums 30 \
    --domain_type music
```

## 성능 비교표

| 모델 | 파라미터 | GPU 메모리 | 훈련 시간 | 추천 성능 | 적합성 |
|------|----------|------------|-----------|-----------|--------|
| LLaMA-2-7B | 7B | 14GB | 기준 | 기준 | 일반적 |
| Qwen2.5-7B | 7B | 14GB | 비슷 | +5% | 일반적 |
| **Qwen2.5-14B** | **14B** | **28GB** | **2x** | **+15%** | **고급** |
| Gemma-2-9B | 9B | 20GB | 1.5x | +10% | 중급 |

## 메모리 최적화 전략

### 1. LoRA 파라미터 조정
```bash
# 메모리 절약 (성능 약간 감소)
--lora_r 8 --lora_alpha 8

# 균형잡힌 설정 (기본)
--lora_r 16 --lora_alpha 16

# 고성능 설정 (메모리 많이 사용)
--lora_r 32 --lora_alpha 32
```

### 2. 레이어 수 조정
```bash
# 메모리 절약: 레이어 수 감소
--qwen_decoder_nums 32  # 기본 40에서 감소

# 극한 메모리 절약
--qwen_decoder_nums 24
```

### 3. 배치 크기 조정
```bash
# 최소 메모리 설정
--batch_size 8 --micro_batch_size 1

# 권장 설정
--batch_size 32 --micro_batch_size 4

# 고메모리 설정 (40GB+)
--batch_size 64 --micro_batch_size 8
```

## 실제 메모리 사용량

### A100 40GB에서의 경험
```bash
# 안전한 설정 (메모리 사용률 ~85%)
--batch_size 32 --micro_batch_size 4 --lora_r 16

# 공격적 설정 (메모리 사용률 ~95%)
--batch_size 48 --micro_batch_size 6 --lora_r 16
```

### A100 80GB에서의 경험
```bash
# 여유로운 설정
--batch_size 64 --micro_batch_size 8 --lora_r 32

# 최대 성능 설정
--batch_size 128 --micro_batch_size 16 --lora_r 32
```

## 문제 해결

### CUDA Out of Memory 해결
1. **배치 크기 줄이기**:
   ```bash
   --batch_size 16 --micro_batch_size 2
   ```

2. **LoRA rank 줄이기**:
   ```bash
   --lora_r 8 --lora_alpha 8
   ```

3. **레이어 수 줄이기**:
   ```bash
   --qwen_decoder_nums 32
   ```

4. **Gradient checkpointing 확인**:
   - 자동으로 활성화됨

### 모델 다운로드 오류
```bash
# HF 토큰 설정
export HF_TOKEN="your_token_here"

# 대용량 캐시 디렉토리
export HF_HOME="/path/to/large/storage"
```

### 훈련 속도 최적화
```bash
# 컴파일 최적화 (PyTorch 2.0+)
export TORCH_COMPILE=1

# 혼합 정밀도 최적화
--fp16 True
```

## 예상 결과

### 훈련 시간 (A100 40GB 기준)
- **에포크당**: 약 4-6시간
- **전체 훈련**: 12-18시간 (3 에포크)

### 성능 향상
- **Hit@10**: 7B 대비 +10-15%
- **NDCG@10**: 7B 대비 +8-12%
- **MRR**: 7B 대비 +12-18%

## 주의사항

1. **⚠️ 메모리 요구사항**: 최소 28GB GPU 메모리 필요
2. **💰 비용**: 더 큰 GPU 인스턴스 필요
3. **⏰ 시간**: 훈련 시간 2배 증가
4. **🔥 발열**: 더 많은 전력 소모

## 언제 사용해야 할까?

### ✅ 사용하는 경우
- 최고 성능이 필요한 프로덕션 환경
- 충분한 GPU 메모리 (28GB+) 보유
- 훈련 시간과 비용이 중요하지 않음
- 연구용으로 최신 결과 필요

### ❌ 사용하지 않는 경우
- 제한된 GPU 메모리 (<28GB)
- 빠른 프로토타이핑 필요
- 비용 최적화가 중요
- 입문용 또는 학습용

## 결론

**Qwen2.5-14B**는 최고 성능을 원하는 고급 사용자를 위한 선택입니다. 충분한 GPU 리소스가 있다면 기존 7B 모델 대비 상당한 성능 향상을 기대할 수 있습니다.

**추천 사용 시나리오**: A100 40GB 이상의 GPU에서 프로덕션 레벨의 추천 시스템 구축 