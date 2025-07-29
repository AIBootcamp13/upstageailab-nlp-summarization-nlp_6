# evaluate_metrics.py

import pandas as pd
from rouge import Rouge
import argparse

def compute_rouge_scores(predictions: list, references: list) -> dict:
    """
    주어진 예측과 정답 리스트에 대해 ROUGE 점수를 계산합니다.
    """
    print("Calculating ROUGE scores...")
    rouge = Rouge()
    # avg=True를 통해 전체 샘플에 대한 평균 점수를 계산합니다.
    scores = rouge.get_scores(predictions, references, avg=True)
    
    # F-1 점수만 추출하여 보기 쉽게 만듭니다.
    result = {key: value['f'] for key, value in scores.items()}
    return result

def main(file_path: str):
    """
    CSV 파일을 읽어 ROUGE 점수를 계산하고 출력합니다.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # 'summary' 컬럼이 있는지 확인
    if 'summary' not in df.columns or 'dialogue' not in df.columns:
        print("Error: The CSV file must contain 'dialogue' and 'summary' columns.")
        return

    # 정답 요약문 (Gold Standard)
    references = df['summary'].tolist()
    
    # 예측 요약문을 모사하기 위해, 원문(dialogue)의 앞 100글자를 사용
    # 이는 Extractive Summarization의 간단한 흉내입니다.
    print("Generating mock predictions by truncating original text...")
    predictions = df['dialogue'].apply(lambda x: x[:100]).tolist()

    # ROUGE 점수 계산
    rouge_results = compute_rouge_scores(predictions, references)

    print("\n--- Evaluation Results ---")
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")
        
    print("\n--- Example Comparison ---")
    for i in range(min(3, len(df))):
        print(f"\n[Sample {i+1}]")
        print(f"  - Original  : {df['dialogue'][i][:200]}...")
        print(f"  - Reference : {references[i]}")
        print(f"  - Prediction: {predictions[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ROUGE scores on a summarization dataset.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file to be evaluated.")
    args = parser.parse_args()
    
    main(args.file_path)