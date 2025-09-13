#!/usr/bin/env python3
"""
Transformers 라이브러리 호환성 체크 스크립트
"""

import sys
try:
    import transformers
    from transformers import TrainingArguments
    print(f"✅ Transformers version: {transformers.__version__}")
    
    # TrainingArguments에서 지원하는 매개변수 확인
    import inspect
    args_signature = inspect.signature(TrainingArguments.__init__)
    params = list(args_signature.parameters.keys())
    
    print(f"📋 TrainingArguments에서 지원하는 주요 매개변수:")
    important_params = [
        'eval_strategy', 'evaluation_strategy', 
        'save_strategy', 'logging_steps', 
        'per_device_train_batch_size', 'per_device_eval_batch_size',
        'gradient_accumulation_steps', 'learning_rate',
        'warmup_steps', 'num_train_epochs', 'max_steps'
    ]
    
    for param in important_params:
        if param in params:
            print(f"  ✅ {param}")
        else:
            print(f"  ❌ {param}")
    
    # 새로운 TrainingArguments 인스턴스 생성 테스트
    try:
        test_args = TrainingArguments(
            output_dir="./test",
            per_device_train_batch_size=1,
            eval_strategy="epoch",  # 새로운 방식
            logging_steps=1,
            num_train_epochs=1,
        )
        print("✅ eval_strategy 매개변수 사용 가능")
    except Exception as e:
        print(f"❌ eval_strategy 사용 불가: {e}")
        
        # 이전 버전 방식 시도
        try:
            test_args = TrainingArguments(
                output_dir="./test",
                per_device_train_batch_size=1,
                evaluation_strategy="epoch",  # 이전 방식
                logging_steps=1,
                num_train_epochs=1,
            )
            print("✅ evaluation_strategy 매개변수 사용 가능 (이전 버전)")
        except Exception as e2:
            print(f"❌ evaluation_strategy도 사용 불가: {e2}")
    
    print("\n🔥 호환성 체크 완료!")
    
except ImportError as e:
    print(f"❌ Transformers 라이브러리를 찾을 수 없습니다: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 예상치 못한 오류: {e}")
    sys.exit(1)

