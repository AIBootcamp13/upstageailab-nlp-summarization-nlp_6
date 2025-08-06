# 통합 메트릭 계산 유틸리티

## 개요
이 모듈은 모든 학습 스크립트에서 일관된 Final Score 기반 메트릭 계산을 제공합니다.

## Final Score 계산 방식
```
Final Score = ROUGE-1 F1 + ROUGE-2 F1 + ROUGE-L F1
```

## 주요 기능

### 1. 통합된 메트릭 계산
- `compute_unified_metrics()`: 핵심 메트릭 계산 함수
- 모든 스크립트에서 동일한 로직 사용
- 응답 부분만 추출하여 정확한 평가

### 2. 다양한 사용 방식 지원
- Hugging Face Trainer 호환
- 기존 코드 호환성 유지
- 빠른 텍스트 기반 계산

### 3. 강건한 오류 처리
- 빈 예측/라벨 처리
- 예외 상황 대응
- 기본값 반환

## 사용 방법

### 기본 사용
```python
from utils.unified_metrics import compute_unified_metrics

# 토큰 ID 배열로 계산
result = compute_unified_metrics(predictions, labels, tokenizer, config)
print(f"Final Score: {result['final_score']}")
```

### Hugging Face Trainer용
```python
from utils.unified_metrics import create_unified_compute_metrics_function

# 메트릭 함수 생성
compute_metrics = create_unified_compute_metrics_function(tokenizer, config)

# Trainer에서 사용
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
    ...
)
```

### 빠른 텍스트 계산
```python
from utils.unified_metrics import quick_rouge_score

scores = quick_rouge_score("예측 텍스트", "정답 텍스트")
print(f"Final Score: {scores['final_score']}")
```

## 반환 메트릭
- `rouge-1`: ROUGE-1 F1 점수
- `rouge-2`: ROUGE-2 F1 점수  
- `rouge-l`: ROUGE-L F1 점수
- `final_score`: 최종 점수 (위 3개의 합)
- `eval_final_score`: Early stopping용 (final_score와 동일)

## 설정 옵션
config 딕셔너리에서 다음 설정을 지원합니다:
```python
config = {
    'inference': {
        'remove_tokens': ['<usr>', '</s>', '<pad>']  # 제거할 토큰들
    }
}
```

## 업데이트된 파일들
- `main.py`: 통합 메트릭 사용
- `experiments/train_with_advanced_data.py`: 통합 메트릭 사용
- `experiments/quick_experiment.py`: 통합 메트릭 사용
- `pipeline/custom_metrics.py`: 통합 메트릭 사용

## 테스트
```bash
python utils/unified_metrics.py
```

이 명령으로 메트릭 계산이 정상 작동하는지 확인할 수 있습니다.
