"""
전처리 결과 비교 분석 스크립트
"""

import pandas as pd
import numpy as np

def compare_datasets():
    """원본, 기본 전처리, 고급 전처리 결과 비교"""
    
    print("=== 전처리 결과 비교 분석 ===\n")
    
    # 데이터 로드
    original_train = pd.read_csv('data/train.csv')
    basic_train = pd.read_csv('processed_data/train_processed.csv')
    advanced_train = pd.read_csv('advanced_processed_data/train_advanced.csv')
    
    datasets = {
        '원본': original_train,
        '기본 전처리': basic_train,
        '고급 전처리': advanced_train
    }
    
    # 기본 통계 비교
    print("1. 데이터셋 크기 비교")
    print("-" * 40)
    for name, df in datasets.items():
        print(f"{name:12}: {len(df):,} 샘플")
    
    print("\n2. 텍스트 길이 통계")
    print("-" * 60)
    print(f"{'데이터셋':12} {'대화 평균':>10} {'대화 중간값':>10} {'요약 평균':>10} {'요약 중간값':>10}")
    print("-" * 60)
    
    for name, df in datasets.items():
        dialogue_mean = df['dialogue'].str.len().mean()
        dialogue_median = df['dialogue'].str.len().median()
        summary_mean = df['summary'].str.len().mean()
        summary_median = df['summary'].str.len().median()
        
        print(f"{name:12} {dialogue_mean:>10.1f} {dialogue_median:>10.1f} {summary_mean:>10.1f} {summary_median:>10.1f}")
    
    print("\n3. 압축 비율 (요약길이/대화길이)")
    print("-" * 40)
    for name, df in datasets.items():
        compression_ratio = df['summary'].str.len().mean() / df['dialogue'].str.len().mean()
        print(f"{name:12}: {compression_ratio:.3f}")
    
    # 샘플 비교
    print("\n4. 전처리 결과 샘플 비교")
    print("=" * 80)
    
    sample_idx = 0
    print(f"\n[샘플 {sample_idx}]")
    
    print("\n원본 대화:")
    print(original_train.iloc[sample_idx]['dialogue'][:200] + "...")
    
    print("\n기본 전처리 대화:")
    print(basic_train.iloc[sample_idx]['dialogue'][:200] + "...")
    
    print("\n고급 전처리 대화:")
    print(advanced_train.iloc[sample_idx]['dialogue'][:200] + "...")
    
    print(f"\n원본 요약: {original_train.iloc[sample_idx]['summary']}")
    print(f"기본 전처리 요약: {basic_train.iloc[sample_idx]['summary']}")
    print(f"고급 전처리 요약: {advanced_train.iloc[sample_idx]['summary']}")
    
    # 토픽 분포 비교
    print("\n5. 토픽 분포 비교")
    print("-" * 40)
    for name, df in datasets.items():
        unique_topics = df['topic'].nunique()
        most_common_topic = df['topic'].value_counts().index[0]
        most_common_count = df['topic'].value_counts().iloc[0]
        
        print(f"{name:12}: {unique_topics:,}개 토픽, 최다 토픽: '{most_common_topic}' ({most_common_count}개)")

def analyze_removed_samples():
    """제거된 샘플 분석"""
    print("\n=== 제거된 샘플 분석 ===")
    
    original_train = pd.read_csv('data/train.csv')
    advanced_train = pd.read_csv('advanced_processed_data/train_advanced.csv')
    
    # 제거된 샘플 찾기
    original_fnames = set(original_train['fname'])
    advanced_fnames = set(advanced_train['fname'])
    removed_fnames = original_fnames - advanced_fnames
    
    print(f"제거된 샘플 수: {len(removed_fnames)}")
    
    if len(removed_fnames) > 0:
        # 제거된 샘플들의 특성 분석
        removed_samples = original_train[original_train['fname'].isin(removed_fnames)]
        
        print(f"제거된 샘플들의 대화 길이 - 평균: {removed_samples['dialogue'].str.len().mean():.1f}")
        print(f"제거된 샘플들의 요약 길이 - 평균: {removed_samples['summary'].str.len().mean():.1f}")
        
        print("\n제거된 샘플 예시:")
        for i, (_, row) in enumerate(removed_samples.head(3).iterrows()):
            print(f"\n[제거된 샘플 {i+1}]")
            print(f"대화 길이: {len(row['dialogue'])}")
            print(f"요약 길이: {len(row['summary'])}")
            print(f"대화: {row['dialogue'][:100]}...")
            print(f"요약: {row['summary']}")

def main():
    compare_datasets()
    analyze_removed_samples()
    
    print("\n=== 전처리 추천사항 ===")
    print("1. 기본 전처리: 단순한 정리만 필요한 경우")
    print("2. 고급 전처리: 모델 학습 성능 향상을 위해 데이터 품질을 높이고 싶은 경우")
    print("3. 고급 전처리에서는 일부 샘플이 제거되므로 데이터 손실을 고려해야 함")
    print("4. 화자 정규화(A, B)는 모델이 화자 패턴을 더 잘 학습할 수 있도록 도움")

if __name__ == "__main__":
    main()