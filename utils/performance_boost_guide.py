"""
성능 향상을 위한 종합 가이드
현재 점수: 34.72 → 목표: 50+ 점
"""

import pandas as pd
import os

def show_performance_roadmap():
    """성능 향상 로드맵"""
    
    print("🎯 성능 향상 로드맵")
    print("현재: 34.72점 → 목표: 50+점 (15+점 향상 필요)")
    print("="*60)
    
    roadmap = [
        {
            "단계": "1단계 - 즉시 개선 (우선순위 최고)",
            "방법": [
                "✅ 더 큰 모델 사용 (KoBART-Large)",
                "✅ 학습률 최적화 (Cosine + Warmup)",
                "✅ Beam Search 파라미터 튜닝",
                "✅ 화자 정보 강화"
            ],
            "예상_향상": "+8-12점",
            "소요_시간": "1-2시간",
            "스크립트": "scripts/quick_train_optimized.py"
        },
        {
            "단계": "2단계 - 앙상블 기법",
            "방법": [
                "🔄 다양한 설정으로 3개 모델 학습",
                "🔄 앙상블 인퍼런스 적용",
                "🔄 투표/가중평균 결합"
            ],
            "예상_향상": "+3-6점",
            "소요_시간": "2-3시간",
            "스크립트": "utils/ensemble_inference.py"
        },
        {
            "단계": "3단계 - 고급 기법",
            "방법": [
                "🧠 데이터 증강 (Back Translation)",
                "🧠 Multi-task Learning",
                "🧠 Curriculum Learning"
            ],
            "예상_향상": "+3-8점",
            "소요_시간": "4-6시간",
            "스크립트": "experiments/advanced_techniques.py"
        }
    ]
    
    for step in roadmap:
        print(f"\n📋 {step['단계']}")
        print(f"   예상 향상: {step['예상_향상']}")
        print(f"   소요 시간: {step['소요_시간']}")
        print(f"   실행 스크립트: {step['스크립트']}")
        print("   방법:")
        for method in step['방법']:
            print(f"     {method}")
    
    print("\n🎯 총 예상 향상: 14-26점")
    print("🏆 목표 달성 점수: 49-61점")

def quick_start_guide():
    """빠른 시작 가이드"""
    
    print("\n🚀 빠른 시작 가이드 (최대 효과)")
    print("="*60)
    
    steps = [
        {
            "순서": "1️⃣",
            "작업": "최적화된 학습 실행",
            "명령어": "python scripts/quick_train_optimized.py",
            "설명": "KoBART-Large + 최적화된 설정으로 학습",
            "예상_시간": "30-60분",
            "예상_효과": "+8-12점"
        },
        {
            "순서": "2️⃣", 
            "작업": "결과 확인 및 제출",
            "명령어": "# optimized_submission.csv 확인",
            "설명": "생성된 결과 파일 검증 후 제출",
            "예상_시간": "5분",
            "예상_효과": "점수 확인"
        },
        {
            "순서": "3️⃣",
            "작업": "추가 모델 학습 (앙상블용)",
            "명령어": "python scripts/quick_train.py  # 다른 설정",
            "설명": "다른 하이퍼파라미터로 추가 모델 학습",
            "예상_시간": "30-60분",
            "예상_효과": "앙상블 준비"
        },
        {
            "순서": "4️⃣",
            "작업": "앙상블 인퍼런스",
            "명령어": "python utils/ensemble_inference.py",
            "설명": "여러 모델의 예측을 결합",
            "예상_시간": "10-20분",
            "예상_효과": "+3-6점"
        }
    ]
    
    for step in steps:
        print(f"\n{step['순서']} {step['작업']}")
        print(f"   명령어: {step['명령어']}")
        print(f"   설명: {step['설명']}")
        print(f"   소요 시간: {step['예상_시간']}")
        print(f"   예상 효과: {step['예상_효과']}")

def troubleshooting_guide():
    """문제 해결 가이드"""
    
    print("\n🔧 문제 해결 가이드")
    print("="*60)
    
    issues = [
        {
            "문제": "GPU 메모리 부족",
            "해결책": [
                "per_device_train_batch_size를 2 → 1로 감소",
                "gradient_accumulation_steps를 8 → 16으로 증가",
                "fp16=True 확인",
                "더 작은 모델 사용 (large → base)"
            ]
        },
        {
            "문제": "학습 속도가 너무 느림",
            "해결책": [
                "데이터 일부만 사용 (train_df.head(5000))",
                "num_train_epochs를 3 → 2로 감소",
                "eval_steps를 300 → 500으로 증가",
                "dataloader_num_workers 증가"
            ]
        },
        {
            "문제": "모델 성능이 향상되지 않음",
            "해결책": [
                "학습률 조정 (5e-5 → 3e-5 또는 1e-4)",
                "더 많은 에폭 학습 (3 → 5)",
                "다른 모델 아키텍처 시도",
                "전처리 방법 변경"
            ]
        },
        {
            "문제": "생성된 요약이 이상함",
            "해결책": [
                "length_penalty 조정 (1.2 → 1.0 또는 1.5)",
                "repetition_penalty 조정 (1.1 → 1.2)",
                "num_beams 조정 (5 → 3 또는 8)",
                "max_length/min_length 조정"
            ]
        }
    ]
    
    for issue in issues:
        print(f"\n❌ {issue['문제']}")
        print("   해결책:")
        for solution in issue['해결책']:
            print(f"     • {solution}")

def check_system_requirements():
    """시스템 요구사항 확인"""
    
    print("\n💻 시스템 요구사항 확인")
    print("="*60)
    
    import torch
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 6:
            print("✅ 메모리 충분: Large 모델 사용 가능")
            print("💡 권장: KoBART-Large 또는 BART-Large")
        else:
            print("⚠️ 메모리 부족: Base 모델 권장")
            print("💡 권장: BART-Base 또는 배치 크기 감소")
    else:
        print("❌ GPU 없음: CPU 학습 (매우 느림)")
        print("💡 권장: Colab 또는 GPU 환경 사용")
    
    # 데이터 확인
    data_files = [
        'insight_processed_data/train_insight.csv',
        'insight_processed_data/dev_insight.csv', 
        'insight_processed_data/test_insight.csv'
    ]
    
    print("\n📊 데이터 파일 확인:")
    for file in data_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"✅ {file}: {len(df):,} 샘플")
        else:
            print(f"❌ {file}: 파일 없음")
    
    # 디스크 공간 확인
    import shutil
    free_space = shutil.disk_usage('.').free / 1024**3
    print(f"\n💾 사용 가능 디스크 공간: {free_space:.1f}GB")
    
    if free_space >= 10:
        print("✅ 디스크 공간 충분")
    else:
        print("⚠️ 디스크 공간 부족: 모델 저장 시 주의")

def generate_experiment_plan():
    """실험 계획 생성"""
    
    print("\n📋 실험 계획")
    print("="*60)
    
    experiments = [
        {
            "실험": "Experiment 1 - 기본 최적화",
            "모델": "gogamza/kobart-large-v2",
            "설정": "lr=5e-5, epochs=3, batch_size=2x8",
            "목표": "현재 대비 +8-12점",
            "우선순위": "🔥 최고"
        },
        {
            "실험": "Experiment 2 - 보수적 학습",
            "모델": "facebook/bart-base",
            "설정": "lr=3e-5, epochs=5, batch_size=4x4",
            "목표": "안정적 성능, 앙상블용",
            "우선순위": "⭐ 높음"
        },
        {
            "실험": "Experiment 3 - 공격적 학습",
            "모델": "gogamza/kobart-large-v2",
            "설정": "lr=1e-4, epochs=2, batch_size=1x16",
            "목표": "빠른 수렴, 높은 성능",
            "우선순위": "💡 중간"
        },
        {
            "실험": "Experiment 4 - T5 아키텍처",
            "모델": "t5-base",
            "설정": "lr=3e-4, epochs=3, batch_size=2x8",
            "목표": "다양성 확보, 앙상블용",
            "우선순위": "💡 중간"
        }
    ]
    
    for exp in experiments:
        print(f"\n{exp['우선순위']} {exp['실험']}")
        print(f"   모델: {exp['모델']}")
        print(f"   설정: {exp['설정']}")
        print(f"   목표: {exp['목표']}")

def main():
    """메인 가이드 함수"""
    
    print("🎯 대화 요약 성능 향상 종합 가이드")
    print("현재 점수: ROUGE-1: 0.4406, Final: 34.72")
    print("목표 점수: 50+ (15+점 향상 필요)")
    print("="*80)
    
    # 시스템 요구사항 확인
    check_system_requirements()
    
    # 성능 향상 로드맵
    show_performance_roadmap()
    
    # 빠른 시작 가이드
    quick_start_guide()
    
    # 실험 계획
    generate_experiment_plan()
    
    # 문제 해결 가이드
    troubleshooting_guide()
    
    print("\n🎉 성공을 위한 핵심 포인트:")
    print("1. 🔥 즉시 실행: python scripts/quick_train_optimized.py")
    print("2. ⭐ 모델 크기가 가장 중요 (base → large)")
    print("3. 💡 앙상블로 추가 향상 (3-6점)")
    print("4. 🎯 목표 달성 가능성: 매우 높음 (49-61점 예상)")
    
    print("\n📞 다음 단계:")
    print("1. 시스템 요구사항 확인 완료")
    print("2. scripts/quick_train_optimized.py 실행")
    print("3. 결과 확인 후 추가 최적화")
    print("4. 필요시 앙상블 기법 적용")

if __name__ == "__main__":
    main()