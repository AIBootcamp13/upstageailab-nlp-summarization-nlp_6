"""
커스텀 메트릭 계산 모듈 (수정된 버전)
Final Score = max_i ROUGE-1-F1(pred, gold_i) + max_i ROUGE-2-F1(pred, gold_i) + max_i ROUGE-L-F1(pred, gold_i)
"""

import numpy as np
from typing import List, Dict, Any
import logging
from .korean_rouge import KoreanRougeCalculator

logger = logging.getLogger(__name__)


class CustomRougeMetrics:
    """커스텀 ROUGE 메트릭 계산기 (한국어 최적화)"""

    def __init__(self):
        # 한국어 최적화 ROUGE 계산기 사용
        self.korean_rouge = KoreanRougeCalculator()

    def compute_rouge_scores(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """단일 예측에 대한 ROUGE 점수 계산"""
        if not references or not prediction.strip():
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }

        # 한국어 최적화 ROUGE 계산 사용
        try:
            scores = self.korean_rouge.calculate_max_rouge_scores(prediction, references)
            return scores
        except Exception as e:
            logger.warning(f"한국어 ROUGE 계산 오류: {e}")
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }

    def compute_final_score(self, prediction: str, references: List[str]) -> float:
        """Final Score 계산"""
        rouge_scores = self.compute_rouge_scores(prediction, references)
        final_score = (
            rouge_scores['rouge1_f1'] +
            rouge_scores['rouge2_f1'] +
            rouge_scores['rougeL_f1']
        )
        return final_score

    def compute_batch_scores(self, predictions: List[str], references_list: List[List[str]]) -> Dict[str, float]:
        """배치 단위 점수 계산"""
        if len(predictions) != len(references_list):
            raise ValueError("예측과 참조 문장의 개수가 일치하지 않습니다.")

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        final_scores = []

        for pred, refs in zip(predictions, references_list):
            scores = self.compute_rouge_scores(pred, refs)
            rouge1_scores.append(scores['rouge1_f1'])
            rouge2_scores.append(scores['rouge2_f1'])
            rougeL_scores.append(scores['rougeL_f1'])

            final_score = scores['rouge1_f1'] + scores['rouge2_f1'] + scores['rougeL_f1']
            final_scores.append(final_score)

        return {
            'rouge1_f1': np.mean(rouge1_scores),
            'rouge2_f1': np.mean(rouge2_scores),
            'rougeL_f1': np.mean(rougeL_scores),
            'final_score': np.mean(final_scores),
            'rouge1_f1_std': np.std(rouge1_scores),
            'rouge2_f1_std': np.std(rouge2_scores),
            'rougeL_f1_std': np.std(rougeL_scores),
            'final_score_std': np.std(final_scores)
        }


def compute_metrics_for_trainer(eval_pred, tokenizer):
    """
    Hugging Face Trainer용 메트릭 계산 함수 (통합된 final_score 버전)
    utils/unified_metrics.py의 통합 메트릭을 사용합니다.

    Args:
        eval_pred: (predictions, labels) 튜플
        tokenizer: 토크나이저

    Returns:
        Dict[str, float]: 계산된 메트릭들
    """
    try:
        # 통합 메트릭 사용
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.unified_metrics import compute_unified_metrics

        predictions, labels = eval_pred
        result = compute_unified_metrics(predictions, labels, tokenizer, verbose=True)

        # 로깅
        logger.info(f"평가 메트릭:")
        logger.info(f"  ROUGE-1 F1: {result['rouge-1']:.4f}")
        logger.info(f"  ROUGE-2 F1: {result['rouge-2']:.4f}")
        logger.info(f"  ROUGE-L F1: {result['rouge-l']:.4f}")
        logger.info(f"  Final Score: {result['final_score']:.4f}")

        return result

    except Exception as e:
        logger.error(f"메트릭 계산 오류: {e}")
        logger.error(f"오류 상세: {type(e).__name__}: {str(e)}")

        # 오류 발생 시 기본값 반환
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
            "final_score": 0.0,
            "eval_final_score": 0.0
        }


def create_compute_metrics_function(tokenizer):
    """토크나이저를 바인딩한 메트릭 계산 함수 생성"""
    def compute_metrics(eval_pred):
        return compute_metrics_for_trainer(eval_pred, tokenizer)
    return compute_metrics
