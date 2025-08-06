#!/usr/bin/env python3
"""
통합된 Final Score 기반 메트릭 계산 유틸리티
모든 학습 스크립트에서 일관된 메트릭 계산을 위한 모듈
"""

import numpy as np
from rouge import Rouge
import logging

logger = logging.getLogger(__name__)

def compute_unified_metrics(predictions, labels, tokenizer, config=None, verbose=True):
    """
    통합된 Final Score 기반 메트릭 계산

    Args:
        predictions: 예측 토큰 ID 배열
        labels: 정답 토큰 ID 배열
        tokenizer: 토크나이저
        config: 설정 딕셔너리 (remove_tokens 포함)
        verbose: 상세 출력 여부

    Returns:
        Dict[str, float]: 계산된 메트릭들
    """
    try:
        # predict_with_generate=True를 사용하면 predictions는 생성된 토큰 ID입니다.
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge-L의 F1 점수를 계산합니다.
        rouge = Rouge()

        # 생성된 텍스트와 레이블에서 응답 부분만 추출하여 비교합니다.
        cleaned_preds = [
            pred.split("### Response:")[1].strip() if "### Response:" in pred else pred.strip()
            for pred in decoded_preds
        ]
        cleaned_labels = [
            label.split("### Response:")[1].strip() if "### Response:" in label else label.strip()
            for label in decoded_labels
        ]

        # 설정에서 제거할 토큰들 가져오기
        if config and 'inference' in config and 'remove_tokens' in config['inference']:
            remove_tokens = config['inference']['remove_tokens']
        else:
            remove_tokens = ['<usr>', '</s>', '<pad>']  # 기본값

        # 불필요한 토큰들을 제거합니다.
        for token in remove_tokens:
            cleaned_preds = [sentence.replace(token, " ") for sentence in cleaned_preds]
            cleaned_labels = [sentence.replace(token, " ") for sentence in cleaned_labels]

        # 비어있는 예측이나 레이블이 있을 경우 점수 계산에서 제외합니다.
        valid_pairs = [(pred, label) for pred, label in zip(cleaned_preds, cleaned_labels)
                       if pred and label]

        if not valid_pairs:
            if verbose:
                logger.warning("유효한 예측-라벨 쌍이 없습니다.")
            return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "final_score": 0, "eval_final_score": 0}

        valid_preds, valid_labels = zip(*valid_pairs)

        # 예측 결과와 정답을 출력합니다.
        if verbose:
            print('-'*150)
            if len(valid_preds) > 0:
                print(f"PRED: {valid_preds[0]}")
                print(f"GOLD: {valid_labels[0]}")
            print('-'*150)
            if len(valid_preds) > 1:
                print(f"PRED: {valid_preds[1]}")
                print(f"GOLD: {valid_labels[1]}")
            print('-'*150)
            if len(valid_preds) > 2:
                print(f"PRED: {valid_preds[2]}")
                print(f"GOLD: {valid_labels[2]}")

        scores = rouge.get_scores(list(valid_preds), list(valid_labels), avg=True)

        result = {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }

        # final_score 계산
        result["final_score"] = result["rouge-1"] + result["rouge-2"] + result["rouge-l"]
        result["eval_final_score"] = result["final_score"]  # early stopping용

        if verbose:
            logger.info(f"Final Score: {result['final_score']:.4f}")

        return result

    except Exception as e:
        logger.error(f"메트릭 계산 오류: {e}")
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "final_score": 0, "eval_final_score": 0}

def create_unified_compute_metrics_function(tokenizer, config=None):
    """
    통합된 메트릭 계산 함수 생성기
    Hugging Face Trainer에서 사용할 수 있는 함수를 반환

    Args:
        tokenizer: 토크나이저
        config: 설정 딕셔너리

    Returns:
        function: compute_metrics 함수
    """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return compute_unified_metrics(predictions, labels, tokenizer, config)

    return compute_metrics

def create_legacy_compute_metrics_function(tokenizer, config):
    """
    기존 스타일 호환성을 위한 함수 생성기

    Args:
        tokenizer: 토크나이저
        config: 설정 딕셔너리

    Returns:
        function: compute_metrics 함수 (기존 스타일)
    """
    def compute_metrics(config_param, tokenizer_param, pred):
        # 기존 스타일에서는 pred.predictions, pred.labels 형태
        if hasattr(pred, 'predictions') and hasattr(pred, 'label_ids'):
            predictions = pred.predictions
            labels = pred.label_ids
        else:
            predictions, labels = pred

        return compute_unified_metrics(predictions, labels, tokenizer, config)

    return compute_metrics

# 편의 함수들
def quick_rouge_score(pred_text, gold_text):
    """
    빠른 ROUGE 점수 계산 (텍스트 입력)

    Args:
        pred_text: 예측 텍스트
        gold_text: 정답 텍스트

    Returns:
        Dict[str, float]: ROUGE 점수들
    """
    try:
        rouge = Rouge()
        scores = rouge.get_scores([pred_text], [gold_text], avg=True)

        result = {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
        result["final_score"] = result["rouge-1"] + result["rouge-2"] + result["rouge-l"]

        return result
    except:
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "final_score": 0}

def batch_rouge_scores(pred_texts, gold_texts):
    """
    배치 ROUGE 점수 계산

    Args:
        pred_texts: 예측 텍스트 리스트
        gold_texts: 정답 텍스트 리스트

    Returns:
        Dict[str, float]: 평균 ROUGE 점수들
    """
    try:
        rouge = Rouge()
        scores = rouge.get_scores(pred_texts, gold_texts, avg=True)

        result = {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
        result["final_score"] = result["rouge-1"] + result["rouge-2"] + result["rouge-l"]

        return result
    except:
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "final_score": 0}

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 통합 메트릭 계산 테스트")

    # 간단한 텍스트 테스트
    pred = "안녕하세요 오늘 날씨가 좋네요"
    gold = "안녕하세요 날씨가 정말 좋습니다"

    scores = quick_rouge_score(pred, gold)
    print(f"\n📊 테스트 결과:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")
    print(f"  final_score: {scores['final_score']:.4f}")
