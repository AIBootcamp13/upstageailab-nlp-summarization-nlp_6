"""
모든 전처리 방법 비교 분석
"""

import pandas as pd
import numpy as np
import os
import re

def compare_all_preprocessing_methods():
    """모든 전처리 방법 비교"""
    
    print("=== 전체 전처리 방법 비교 분석 ===\n")
    
    # 데이터 로드
    datasets = {}
    
    # 원본 데이터
    if os.path.exists('data/train.csv'):
        datasets['원본'] = pd.read_csv('data/train.csv')
    
    # 기본 전처리
    if os.path.exists('processed_data/train_processed.csv'):
        datasets['기본 전처리'] = pd.read_csv('processed_data/train_processed.csv')
    
    # 고급 전처리
    if os.path.exists('advanced_processed_data/train_advanced.csv'):
        datasets['고급 전처리'] = pd.read_csv('advanced_processed_data/train_advanced.csv')
    
    # 향상된 전처리
    if os.path.exists('enhanced_processed_data/train_enhanced.csv'):
        datasets['향상된 전처리'] = pd.read_csv('enhanced_processed_data/train_enhanced.csv')
    
    if not datasets:
        print("전처리된 데이터를 찾을 수 없습니다.")
        return
    
    # 1. 데이터셋 크기 비교
    print("1. 데이터셋 크기 비교")
    print("-" * 50)
    print(f"{'방법':15} {'샘플 수':>10} {'보존율':>10}")
    print("-" * 50)
    
    original_size = len(datasets['원본']) if '원본' in datasets else 0
    
    for name, df in datasets.items():
        size = len(df)
        retention_rate = (size / original_size * 100) if original_size > 0 else 100
        print(f"{name:15} {size:>10,} {retention_rate:>9.1f}%")
    
    # 2. 텍스트 길이 통계
    print(f"\n2. 텍스트 길이 통계")
    print("-" * 80)
    print(f"{'방법':15} {'대화 평균':>10} {'대화 중간값':>10} {'요약 평균':>10} {'요약 중간값':>10} {'압축비율':>10}")
    print("-" * 80)
    
    for name, df in datasets.items():
        dialogue_mean = df['dialogue'].str.len().mean()
        dialogue_median = df['dialogue'].str.len().median()
        summary_mean = df['summary'].str.len().mean()
        summary_median = df['summary'].str.len().median()
        compression_ratio = summary_mean / dialogue_mean
        
        print(f"{name:15} {dialogue_mean:>10.1f} {dialogue_median:>10.1f} {summary_mean:>10.1f} {summary_median:>10.1f} {compression_ratio:>10.3f}")
    
    # 3. 화자 태그 분석
    print(f"\n3. 화자 태그 분석")
    print("-" * 60)
    print(f"{'방법':15} {'Person1 태그':>12} {'Person2 태그':>12} {'A: 태그':>10} {'B: 태그':>10}")
    print("-" * 60)
    
    for name, df in datasets.items():
        person1_count = df['dialogue'].str.contains('#Person1#:|Person1:', regex=True).sum()
        person2_count = df['dialogue'].str.contains('#Person2#:|Person2:', regex=True).sum()
        a_count = df['dialogue'].str.contains('A:', regex=False).sum()
        b_count = df['dialogue'].str.contains('B:', regex=False).sum()
        
        print(f"{name:15} {person1_count:>12} {person2_count:>12} {a_count:>10} {b_count:>10}")
    
    # 4. 특수 토큰 사용 현황
    print(f"\n4. 특수 토큰 사용 현황")
    print("-" * 70)
    print(f"{'방법':15} {'PhoneNumber':>12} {'Address':>10} {'Email':>8} {'CardNumber':>12}")
    print("-" * 70)
    
    for name, df in datasets.items():
        phone_count = df['dialogue'].str.contains('#PhoneNumber#', regex=False).sum()
        address_count = df['dialogue'].str.contains('#Address#', regex=False).sum()
        email_count = df['dialogue'].str.contains('#Email#', regex=False).sum()
        card_count = df['dialogue'].str.contains('#CardNumber#', regex=False).sum()
        
        print(f"{name:15} {phone_count:>12} {address_count:>10} {email_count:>8} {card_count:>12}")
    
    # 5. 샘플 비교
    print(f"\n5. 전처리 결과 샘플 비교")
    print("=" * 100)
    
    sample_idx = 0
    for name, df in datasets.items():
        if len(df) > sample_idx:
            print(f"\n[{name}]")
            print(f"대화: {df.iloc[sample_idx]['dialogue'][:150]}...")
            print(f"요약: {df.iloc[sample_idx]['summary']}")
    
    # 6. 품질 지표 분석
    print(f"\n6. 데이터 품질 지표")
    print("-" * 80)
    print(f"{'방법':15} {'평균 턴수':>10} {'반복패턴':>10} {'긴 공백':>10} {'구두점 누락':>12}")
    print("-" * 80)
    
    for name, df in datasets.items():
        # 평균 턴 수
        avg_turns = df['dialogue'].apply(lambda x: len(re.findall(r'(A:|B:|#Person\d+#:)', x))).mean()
        
        # 반복 패턴 (동일 문자 3회 이상)
        repeat_pattern = df['dialogue'].str.contains(r'(.)\1{3,}', regex=True).sum()
        
        # 긴 공백 (3개 이상)
        long_spaces = df['dialogue'].str.contains(r'\s{3,}', regex=True).sum()
        
        # 구두점 누락 (문장 끝에 구두점 없음)
        missing_punct = df['dialogue'].apply(
            lambda x: 1 if x and x.strip()[-1] not in '.!?' else 0
        ).sum()
        
        print(f"{name:15} {avg_turns:>10.1f} {repeat_pattern:>10} {long_spaces:>10} {missing_punct:>12}")

def analyze_removed_samples():
    """제거된 샘플들의 특성 분석"""
    print(f"\n=== 제거된 샘플 특성 분석 ===")
    
    # 원본과 각 전처리 결과 비교
    original = pd.read_csv('data/train.csv')
    
    preprocessing_methods = [
        ('기본 전처리', 'processed_data/train_processed.csv'),
        ('고급 전처리', 'advanced_processed_data/train_advanced.csv'),
        ('향상된 전처리', 'enhanced_processed_data/train_enhanced.csv')
    ]
    
    original_fnames = set(original['fname'])
    
    for method_name, file_path in preprocessing_methods:
        if os.path.exists(file_path):
            processed = pd.read_csv(file_path)
            processed_fnames = set(processed['fname'])
            removed_fnames = original_fnames - processed_fnames
            
            if removed_fnames:
                removed_samples = original[original['fname'].isin(removed_fnames)]
                
                print(f"\n{method_name}에서 제거된 샘플 ({len(removed_samples)}개):")
                print(f"  - 평균 대화 길이: {removed_samples['dialogue'].str.len().mean():.1f}자")
                print(f"  - 평균 요약 길이: {removed_samples['summary'].str.len().mean():.1f}자")
                
                # 턴 수 분석
                removed_samples['turn_count'] = removed_samples['dialogue'].apply(
                    lambda x: len(re.findall(r'#Person\d+#:', x))
                )
                print(f"  - 평균 턴 수: {removed_samples['turn_count'].mean():.1f}")
                print(f"  - 턴 수 범위: {removed_samples['turn_count'].min()} ~ {removed_samples['turn_count'].max()}")

def generate_recommendations():
    """전처리 방법별 추천사항"""
    print(f"\n=== 전처리 방법별 추천사항 ===")
    
    recommendations = {
        "기본 전처리": {
            "장점": ["데이터 손실 없음", "빠른 처리 속도", "안정성"],
            "단점": ["품질 개선 제한적", "노이즈 데이터 포함"],
            "추천 상황": ["전체 데이터 보존이 중요한 경우", "빠른 프로토타이핑", "베이스라인 구축"]
        },
        "고급 전처리": {
            "장점": ["화자 정규화", "길이 기반 품질 향상", "적당한 데이터 손실"],
            "단점": ["일부 데이터 손실", "처리 시간 증가"],
            "추천 상황": ["모델 성능 향상 목적", "화자 패턴 학습 중요", "균형잡힌 품질-양 트레이드오프"]
        },
        "향상된 전처리": {
            "장점": ["최고 품질", "다중 화자 처리", "개인정보 보호 강화", "노이즈 제거"],
            "단점": ["상당한 데이터 손실", "복잡한 처리 과정"],
            "추천 상황": ["최고 성능 추구", "프로덕션 환경", "개인정보 보호 중요", "충분한 데이터 보유"]
        }
    }
    
    for method, info in recommendations.items():
        print(f"\n{method}:")
        print(f"  장점: {', '.join(info['장점'])}")
        print(f"  단점: {', '.join(info['단점'])}")
        print(f"  추천 상황: {', '.join(info['추천 상황'])}")

def main():
    import re
    
    compare_all_preprocessing_methods()
    analyze_removed_samples()
    generate_recommendations()
    
    print(f"\n" + "="*80)
    print("결론:")
    print("1. 기본 전처리: 안전하고 빠른 선택")
    print("2. 고급 전처리: 성능과 안정성의 균형")
    print("3. 향상된 전처리: 최고 품질, 하지만 데이터 손실 고려 필요")
    print("="*80)

if __name__ == "__main__":
    main()