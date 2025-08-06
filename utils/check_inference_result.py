"""
추론 결과 검증 스크립트
"""

import pandas as pd

def check_inference_result():
    """추론 결과 검증"""
    
    # 파일 로드
    result_df = pd.read_csv('./prediction/output.csv')
    original_df = pd.read_csv('data/test.csv')
    
    print(f"📊 추론 결과: {len(result_df)} 샘플")
    print(f"📊 원본 테스트: {len(original_df)} 샘플")
    
    # 누락된 샘플 확인
    original_fnames = set(original_df['fname'])
    result_fnames = set(result_df['fname'])
    
    missing = original_fnames - result_fnames
    extra = result_fnames - original_fnames
    
    print(f"\n✅ 누락된 샘플: {len(missing)}개")
    if missing:
        print(f"   누락 목록: {sorted(list(missing))}")
    
    print(f"✅ 추가된 샘플: {len(extra)}개")
    if extra:
        print(f"   추가 목록: {sorted(list(extra))}")
    
    # 특정 샘플 확인
    test_samples = ['test_87', 'test_434', 'test_0', 'test_498']
    print(f"\n🔍 특정 샘플 확인:")
    for sample in test_samples:
        included = sample in result_fnames
        print(f"   {sample}: {'✅ 포함' if included else '❌ 누락'}")
        
        if included:
            summary = result_df[result_df['fname'] == sample]['summary'].iloc[0]
            print(f"      요약: {summary[:50]}...")
    
    # 순서 확인
    if len(result_df) == len(original_df) and missing == set() and extra == set():
        print(f"\n🎉 완벽! 모든 테스트 샘플이 포함되었습니다!")
        
        # 순서 확인
        original_order = original_df['fname'].tolist()
        result_order = result_df['fname'].tolist()
        
        if original_order == result_order:
            print("✅ 순서도 정확합니다!")
        else:
            print("⚠️ 순서가 다릅니다.")
    else:
        print(f"\n⚠️ 일부 문제가 있습니다.")

if __name__ == "__main__":
    check_inference_result()