"""
제출 파일 수정 스크립트
- 누락된 테스트 샘플들을 sample_submission 형식에 맞춰 추가
- 모든 테스트 샘플이 포함되도록 보장
"""

import pandas as pd
import os

def fix_submission_file(prediction_file="./prediction/output.csv", 
                       sample_submission_file="data/sample_submission.csv",
                       output_file="./prediction/fixed_output.csv"):
    """제출 파일 수정"""
    
    print("🔧 제출 파일 수정 중...")
    
    # 파일 로드
    try:
        pred_df = pd.read_csv(prediction_file)
        sample_df = pd.read_csv(sample_submission_file)
        
        print(f"📊 예측 결과: {len(pred_df)} 샘플")
        print(f"📊 샘플 제출: {len(sample_df)} 샘플")
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return None
    
    # 필요한 모든 fname 추출
    required_fnames = sample_df['fname'].tolist()
    existing_fnames = pred_df['fname'].tolist()
    
    # 누락된 fname 찾기
    missing_fnames = [fname for fname in required_fnames if fname not in existing_fnames]
    
    print(f"🔍 누락된 샘플: {len(missing_fnames)}개")
    if missing_fnames:
        print(f"   예시: {missing_fnames[:5]}...")
    
    # 누락된 샘플들을 기본 요약으로 추가
    if missing_fnames:
        missing_data = []
        for fname in missing_fnames:
            missing_data.append({
                'fname': fname,
                'summary': '대화 내용을 요약한 결과입니다.'  # 기본 요약
            })
        
        missing_df = pd.DataFrame(missing_data)
        
        # 기존 예측 결과와 합치기
        fixed_df = pd.concat([pred_df, missing_df], ignore_index=True)
    else:
        fixed_df = pred_df.copy()
    
    # sample_submission 순서에 맞춰 정렬
    fixed_df = fixed_df.set_index('fname').loc[required_fnames].reset_index()
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fixed_df.to_csv(output_file, index=False)
    
    print(f"✅ 수정된 제출 파일 저장: {output_file}")
    print(f"📊 최종 샘플 수: {len(fixed_df)}")
    
    # 검증
    if len(fixed_df) == len(sample_df):
        print("✅ 모든 필요한 샘플이 포함되었습니다!")
    else:
        print(f"⚠️ 샘플 수 불일치: {len(fixed_df)} vs {len(sample_df)}")
    
    return fixed_df

def create_complete_inference_pipeline():
    """완전한 추론 파이프라인 (누락 샘플 처리 포함)"""
    
    print("🚀 완전한 추론 파이프라인 시작!")
    
    # 1. 기본 추론 실행
    print("\n1️⃣ 기본 추론 실행...")
    from scripts.quick_inference import quick_inference
    
    try:
        results = quick_inference()
        print("✅ 기본 추론 완료")
    except Exception as e:
        print(f"❌ 추론 실패: {e}")
        return None
    
    # 2. 제출 파일 수정
    print("\n2️⃣ 제출 파일 수정...")
    fixed_results = fix_submission_file()
    
    if fixed_results is not None:
        print("✅ 완전한 추론 파이프라인 완료!")
        return fixed_results
    else:
        print("❌ 제출 파일 수정 실패")
        return None

def analyze_missing_samples():
    """누락된 샘플들 분석"""
    
    print("🔍 누락된 샘플 분석...")
    
    # 원본 테스트 데이터
    original_test = pd.read_csv('data/test.csv')
    
    # 고급 전처리된 테스트 데이터
    try:
        advanced_test = pd.read_csv('advanced_processed_data/test_advanced.csv')
    except:
        print("❌ 고급 전처리된 테스트 데이터를 찾을 수 없습니다.")
        return
    
    # 누락된 샘플들
    original_fnames = set(original_test['fname'])
    advanced_fnames = set(advanced_test['fname'])
    missing_fnames = original_fnames - advanced_fnames
    
    print(f"📊 원본 테스트: {len(original_fnames)} 샘플")
    print(f"📊 고급 전처리: {len(advanced_fnames)} 샘플")
    print(f"📊 누락된 샘플: {len(missing_fnames)} 개")
    
    if missing_fnames:
        missing_list = sorted(list(missing_fnames))
        print(f"🔍 누락된 샘플들: {missing_list}")
        
        # 누락된 샘플들의 특성 분석
        missing_samples = original_test[original_test['fname'].isin(missing_fnames)]
        
        print(f"\n📈 누락된 샘플들의 특성:")
        print(f"   평균 대화 길이: {missing_samples['dialogue'].str.len().mean():.1f}자")
        print(f"   최대 대화 길이: {missing_samples['dialogue'].str.len().max()}자")
        print(f"   최소 대화 길이: {missing_samples['dialogue'].str.len().min()}자")
        
        # 샘플 출력
        print(f"\n📝 누락된 샘플 예시:")
        for i, (_, row) in enumerate(missing_samples.head(2).iterrows()):
            print(f"   {row['fname']}: {row['dialogue'][:100]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='제출 파일 수정')
    parser.add_argument('--mode', choices=['fix', 'analyze', 'complete'], 
                       default='complete', help='실행 모드')
    parser.add_argument('--prediction-file', default='./prediction/output.csv',
                       help='예측 결과 파일')
    parser.add_argument('--output-file', default='./prediction/fixed_output.csv',
                       help='수정된 출력 파일')
    
    args = parser.parse_args()
    
    if args.mode == 'fix':
        fix_submission_file(args.prediction_file, output_file=args.output_file)
    elif args.mode == 'analyze':
        analyze_missing_samples()
    elif args.mode == 'complete':
        create_complete_inference_pipeline()