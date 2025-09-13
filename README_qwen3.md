# SLMRec with Qwen3 Support

이 버전은 기존 LLaMA 기반의 SLMRec을 Qwen3 모델로 변경한 단일 GPU 지원 버전입니다.

## 주요 변경사항

### 1. 모델 아키텍처
- **기존**: LLaMA-2-7B 기반
- **변경**: Qwen2.5-7B 기반 (`Qwen/Qwen2.5-7B`)

### 2. GPU 지원
- **기존**: 멀티 GPU (DDP) 지원
- **변경**: 단일 GPU 최적화

### 3. 새로운 파일들
- `model_qwen3.py`: Qwen3 기반 모델 클래스들
- `finetune_qwen3.py`: 단일 GPU용 fine-tuning 스크립트
- `run_finetune_qwen3.sh`: 실행 스크립트
- `requirements_qwen3.txt`: Qwen3 관련 의존성 포함

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python3 -m venv qwen3_env
source qwen3_env/bin/activate  # Linux/Mac
```

2. 의존성 설치:
```bash
pip install -r requirements_qwen3.txt
```

## 사용법

### 1. 데이터 준비
기존과 동일하게 데이터를 준비합니다:
```bash
python train_sr_trad.py
python extract_emb.py
```

### 2. Teacher 모델 훈련 (Qwen3 기반)
```bash
chmod +x run_finetune_qwen3.sh
bash run_finetune_qwen3.sh ./output/qwen3_teacher_model/
```

### 3. 직접 실행
```bash
python finetune_qwen3.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir ./output/qwen3_teacher/ \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 0.001 \
    --qwen_decoder_nums 32 \
    --domain_type music
```

## 주요 매개변수

### Qwen3 전용 매개변수
- `--base_model`: `"Qwen/Qwen3-7B"` (기본값)
- `--qwen_decoder_nums`: Qwen3 디코더 레이어 수 (기본값: 32)
- `--lora_target_modules`: Qwen3에 맞는 LoRA 타겟 모듈들

### 단일 GPU 최적화
- `--batch_size`: 64 (멀티 GPU 버전보다 감소)
- `--micro_batch_size`: 8
- `--per_device_eval_batch_size`: 256 (자동 설정)
- `--dataloader_num_workers`: 4 (자동 설정)

## 모델 클래스 구조

### 1. LLM4RecQwen3
- 기본 Qwen3 기반 추천 모델
- LLaMA 대신 Qwen2Model 사용
- AutoTokenizer로 토크나이저 초기화

### 2. LLM4RecQwen3Teacher
- Teacher 모델용 클래스
- Hidden states 출력 지원

### 3. LLM4RecQwen3Student
- Student 모델용 클래스 (Knowledge Distillation용)
- Multiple supervision 지원

## 성능 최적화

### 메모리 사용량
- float16 정밀도 사용
- LoRA를 통한 parameter-efficient fine-tuning
- 단일 GPU에 최적화된 배치 크기

### 속도 최적화
- 불필요한 DDP 코드 제거
- 단일 GPU에 맞는 데이터로더 설정
- 효율적인 gradient accumulation

## 주의사항

1. **GPU 메모리**: Qwen2.5-7B는 약 14GB의 GPU 메모리가 필요합니다.
2. **토크나이저**: Qwen3는 자체 토크나이저를 사용하므로 기존 LLaMA 토크나이저와 다를 수 있습니다.
3. **LoRA 모듈**: Qwen3의 아키텍처에 맞게 LoRA target modules가 조정되었습니다.

## 문제 해결

### CUDA 메모리 부족
```bash
# 배치 크기 감소
--batch_size 32 --micro_batch_size 4
```

### 특정 GPU 사용
```bash
export CUDA_VISIBLE_DEVICES=0
bash run_finetune_qwen3.sh ./output/qwen3_teacher/
```

### 모델 다운로드 문제
```bash
# Hugging Face 캐시 디렉토리 설정
export HF_HOME=/path/to/cache
```

## 예상 성능

- **훈련 속도**: 단일 GPU에서 약 2-3x 빠른 처리 (DDP 오버헤드 제거)
- **메모리 효율성**: LoRA + float16으로 메모리 사용량 최적화
- **추천 성능**: 기존 LLaMA 버전과 유사한 성능 예상

## 향후 계획

- [ ] Knowledge Distillation 지원 (Qwen3 Teacher → Qwen3 Student)
- [ ] 더 작은 Qwen 모델 지원 (Qwen2.5-3B, Qwen2.5-1.5B)
- [ ] 추가 도메인 지원 (영화, 책 등) 