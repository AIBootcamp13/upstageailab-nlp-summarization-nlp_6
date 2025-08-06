"""
인사이트 기반 전처리 데이터 확인
"""

import pandas as pd

def check_insight_processed_data():
    """인사이트 기반 전처리 데이터 확인"""
    
    print("📊 인사이트 기반 전처리 데이터 확인")
    print("="*50)
    
    # 각 파일 확인
    files = ['train_insight.csv', 'dev_insight.csv', 'test_insight.csv']
    
    for file in files:
        df = pd.read_csv(f'insight_processed_data/{file}')
        
        print(f"\n📁 {file}:")
        print(f"  샘플 수: {len(df):,}개")
        print(f"  컬럼: {df.columns.tolist()}")
        
        # 첫 번째 샘플 확인
        print(f"  첫 번째 샘플:")
        print(f"    대화: {df.iloc[0]['dialogue'][:100]}...")
        if 'summary' in df.columns:
            print(f"    요약: {df.iloc[0]['summary']}")
        
        # 데이터 품질 확인
        print(f"  데이터 품질:")
        print(f"    대화 평균 길이: {df['dialogue'].str.len().mean():.1f}자")
        if 'summary' in df.columns:
            print(f"    요약 평균 길이: {df['summary'].str.len().mean():.1f}자")
        
        # 인사이트 적용 효과 확인
        colloquial_count = df['dialogue'].str.contains('거야|거예요').sum()
        interjection_count = df['dialogue'].str.contains('정말|너무').sum()
        multi_speaker_count = df['dialogue'].str.contains('#Person[3-9]#').sum()
        
        print(f"  인사이트 적용 효과:")
        print(f"    구어체 표현 잔존: {colloquial_count}개")
        print(f"    감탄사 잔존: {interjection_count}개")
        print(f"    다중 화자 잔존: {multi_speaker_count}개")

if __name__ == "__main__":
    check_insight_processed_data()