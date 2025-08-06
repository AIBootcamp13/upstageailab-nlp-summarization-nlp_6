"""
인사이트 기반 전처리 테스트 (소량 데이터)
"""

import pandas as pd
import sys
import os
sys.path.append('preprocessing')

from insight_based_preprocess import InsightBasedPreprocessor

def test_insight_preprocessing():
    """인사이트 기반 전처리 테스트"""
    
    print("🧪 인사이트 기반 전처리 테스트")
    print("="*50)
    
    # 소량 데이터로 테스트
    train_df = pd.read_csv('data/train.csv').head(100)  # 100개만 테스트
    
    print(f"테스트 데이터: {len(train_df)} 샘플")
    
    # 전처리기 설정
    config = {
        'remove_colloquial': True,
        'remove_interjections': True,
        'preserve_important_keywords': True,
        'normalize_multi_speakers': True,
    }
    
    preprocessor = InsightBasedPreprocessor(config)
    
    # 전처리 실행
    processed_df = preprocessor.preprocess_dataframe(train_df)
    
    print(f"전처리 후: {len(processed_df)} 샘플")
    
    # 결과 비교
    print("\n📊 전처리 전후 비교:")
    print("원본:")
    print(f"  대화: {train_df.iloc[0]['dialogue'][:100]}...")
    print(f"  요약: {train_df.iloc[0]['summary']}")
    
    print("\n전처리 후:")
    print(f"  대화: {processed_df.iloc[0]['dialogue'][:100]}...")
    print(f"  요약: {processed_df.iloc[0]['summary']}")
    
    # 워드클라우드 인사이트 적용 효과 확인
    print("\n🎯 인사이트 적용 효과:")
    
    # 구어체 제거 확인
    original_colloquial = sum([train_df['dialogue'].str.contains(word).sum() for word in ['거야', '거예요']])
    processed_colloquial = sum([processed_df['dialogue'].str.contains(word).sum() for word in ['거야', '거예요']])
    
    print(f"구어체 표현 제거: {original_colloquial} -> {processed_colloquial}")
    
    # 감탄사 제거 확인
    original_interjection = sum([train_df['dialogue'].str.contains(word).sum() for word in ['정말', '너무']])
    processed_interjection = sum([processed_df['dialogue'].str.contains(word).sum() for word in ['정말', '너무']])
    
    print(f"감탄사 제거: {original_interjection} -> {processed_interjection}")
    
    # 다중 화자 정규화 확인
    original_multi = train_df['dialogue'].str.contains('#Person[3-9]#').sum()
    processed_multi = processed_df['dialogue'].str.contains('#Person[3-9]#').sum()
    
    print(f"다중 화자 정규화: {original_multi} -> {processed_multi}")
    
    print("\n✅ 인사이트 기반 전처리 테스트 완료!")
    
    return processed_df

if __name__ == "__main__":
    test_insight_preprocessing()