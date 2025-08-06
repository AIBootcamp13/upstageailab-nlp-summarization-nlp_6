"""
인사이트 기반 전처리 전체 실행 (로그 최소화)
"""

import pandas as pd
import sys
import os
sys.path.append('preprocessing')

from insight_based_preprocess import InsightBasedPreprocessor

def run_full_insight_preprocessing():
    """전체 데이터로 인사이트 기반 전처리 실행"""
    
    print("🚀 인사이트 기반 전처리 전체 실행")
    print("="*50)
    
    # 전처리기 설정 (로그 최소화)
    config = {
        'remove_colloquial': True,
        'remove_interjections': True,
        'preserve_important_keywords': True,
        'normalize_multi_speakers': True,
        'verbose': False  # 로그 최소화
    }
    
    # 전처리기 초기화
    preprocessor = InsightBasedPreprocessor(config)
    
    # 데이터 로드
    print("📊 데이터 로드 중...")
    train_df = pd.read_csv('data/train.csv')
    dev_df = pd.read_csv('data/dev.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"  Train: {len(train_df):,} 샘플")
    print(f"  Dev: {len(dev_df):,} 샘플")
    print(f"  Test: {len(test_df):,} 샘플")
    
    # 전처리 실행 (로그 최소화를 위해 개별 실행)
    print("\n🔧 Train 데이터 전처리 중...")
    train_processed = preprocessor.preprocess_dataframe(train_df)
    
    print("🔧 Dev 데이터 전처리 중...")
    dev_processed = preprocessor.preprocess_dataframe(dev_df)
    
    print("🔧 Test 데이터 전처리 중...")
    test_processed = preprocessor.preprocess_dataframe(test_df, is_test=True)
    
    # 결과 폴더 생성 및 저장
    print("\n💾 결과 저장 중...")
    os.makedirs('insight_processed_data', exist_ok=True)
    
    train_processed.to_csv('insight_processed_data/train_insight.csv', index=False)
    dev_processed.to_csv('insight_processed_data/dev_insight.csv', index=False)
    test_processed.to_csv('insight_processed_data/test_insight.csv', index=False)
    
    print("✅ 인사이트 기반 전처리 완료!")
    print(f"\n📁 저장된 파일:")
    print(f"  - insight_processed_data/train_insight.csv ({len(train_processed):,} 샘플)")
    print(f"  - insight_processed_data/dev_insight.csv ({len(dev_processed):,} 샘플)")
    print(f"  - insight_processed_data/test_insight.csv ({len(test_processed):,} 샘플)")
    
    # 전처리 효과 요약
    print(f"\n📊 전처리 효과 요약:")
    print(f"  Train: {len(train_df):,} → {len(train_processed):,} ({len(train_processed)/len(train_df)*100:.1f}% 보존)")
    print(f"  Dev: {len(dev_df):,} → {len(dev_processed):,} ({len(dev_processed)/len(dev_df)*100:.1f}% 보존)")
    print(f"  Test: {len(test_df):,} → {len(test_processed):,} ({len(test_processed)/len(test_df)*100:.1f}% 보존)")
    
    return train_processed, dev_processed, test_processed

if __name__ == "__main__":
    run_full_insight_preprocessing()