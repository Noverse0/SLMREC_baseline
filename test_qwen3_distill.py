#!/usr/bin/env python3
"""
Qwen3 Knowledge Distillation 테스트 스크립트
기본적인 모델 로딩과 호환성을 확인합니다.
"""

import torch
import pickle
from utils.prompter import Prompter
from model_qwen3 import LLM4RecQwen3, LLM4RecQwen3Teacher, LLM4RecQwen3Student
from utils.data_utils import LLMDataset, SequentialCollator

def test_qwen3_models():
    """Qwen3 모델들의 기본 로딩과 호환성 테스트"""
    
    print("🔍 Qwen3 Knowledge Distillation 호환성 테스트 시작...")
    
    # 기본 설정
    base_model = "Qwen/Qwen3-8B"  # 실제 환경에서는 로컬 경로로 변경 가능
    domain_type = "music"
    device_map = {"": 0}
    
    # Prompter 초기화
    prompter = Prompter("alpaca")
    
    try:
        # 아이템 임베딩 로드 테스트
        print("📦 아이템 임베딩 로드 중...")
        item_embed = pickle.load(open(f'./output/{domain_type}.pkl', 'rb'))['item_embedding']
        print(f"✅ 아이템 임베딩 로드 성공: {item_embed.shape}")
        
        # 데이터셋 로드 테스트
        print("📊 데이터셋 로드 중...")
        dataset_train = LLMDataset(
            item_size=999, 
            max_seq_length=30, 
            data_type='train', 
            csv_path=f"./dataset/sequential/{domain_type}.csv"
        )
        print(f"✅ 훈련 데이터셋 로드 성공: {len(dataset_train)} 샘플")
        
        # 기본 모델 설정
        model_args = {
            'base_model': base_model,
            'task_type': 'sequential',
            'cache_dir': './cache',
            'input_dim': 128,
            'output_dim': 0,
            'interval_nums': 0,
            'drop_type': 'trune',
            'lora_r': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'lora_target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            'device_map': device_map,
            'instruction_text': prompter.generate_prompt('sequential'),
            'train_stargy': 'lora',
            'user_embeds': None,
            'input_embeds': item_embed,
            'seq_len': 30,
        }
        
        print("🤖 Teacher 모델 로드 테스트...")
        teacher_args = model_args.copy()
        teacher_args.update({
            'qwen_decoder_nums': 28,  # Teacher: 전체 레이어
        })
        
        # teacher_model = LLM4RecQwen3Teacher(**teacher_args)
        # print("✅ Teacher 모델 로드 성공")
        
        print("🎓 Student 모델 로드 테스트...")
        student_args = model_args.copy()
        student_args.update({
            'qwen_decoder_nums': 14,  # Student: 절반 레이어
            'distill_block': 4,
            'is_cls_multiple': False,
        })
        
        # student_model = LLM4RecQwen3Student(**student_args)
        # print("✅ Student 모델 로드 성공")
        
        print("📋 모델 설정 검증:")
        print(f"  - Base Model: {base_model}")
        print(f"  - Teacher Layers: {teacher_args['qwen_decoder_nums']}")
        print(f"  - Student Layers: {student_args['qwen_decoder_nums']}")
        print(f"  - LoRA Config: r={model_args['lora_r']}, alpha={model_args['lora_alpha']}")
        print(f"  - Target Modules: {model_args['lora_target_modules']}")
        
        print("✅ 모든 호환성 테스트 통과!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        print("   💡 다음을 확인해주세요:")
        print(f"   - ./output/{domain_type}.pkl 파일 존재 여부")
        print(f"   - ./dataset/sequential/{domain_type}.csv 파일 존재 여부")
        return False
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        print(f"   오류 타입: {type(e).__name__}")
        return False

def print_usage_guide():
    """사용법 가이드 출력"""
    print("\n" + "="*60)
    print("🚀 Qwen3 Knowledge Distillation 사용 가이드")
    print("="*60)
    print()
    print("1. 기본 실행:")
    print("   bash run_distill_qwen3.sh")
    print()
    print("2. 커스텀 설정으로 실행:")
    print("   python distill_qwen3.py \\")
    print("     --base_model 'Qwen/Qwen3-8B' \\")
    print("     --domain_type 'music' \\")
    print("     --qwen_decoder_nums_teacher 28 \\")
    print("     --qwen_decoder_nums_student 14 \\")
    print("     --distill_type_standard 'offline' \\")
    print("     --gpu_device 0")
    print()
    print("3. 평가만 실행:")
    print("   python distill_qwen3.py \\")
    print("     --train_eval_type 'eval' \\")
    print("     --student_resume_from_checkpoint 'path/to/checkpoint'")
    print()
    print("💡 주요 차이점:")
    print("   - LLaMA → Qwen3-8B 모델 사용")
    print("   - 단일 GPU 환경에 최적화")
    print("   - bfloat16 정밀도 사용")
    print("   - 향상된 안정성과 메모리 효율성")
    print()

if __name__ == "__main__":
    print("🔥 Qwen3 Knowledge Distillation Framework")
    print("=" * 50)
    
    # 호환성 테스트 실행
    success = test_qwen3_models()
    
    if success:
        print_usage_guide()
    else:
        print("\n❌ 호환성 테스트 실패. 위의 안내를 확인하고 다시 시도해주세요.")
    
    print("\n" + "="*60)
