"""
한국어 최적화 ROUGE 계산 모듈
"""

import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np


class KoreanRougeCalculator:
    """한국어 최적화 ROUGE 계산기"""

    def __init__(self):
        pass

    def _tokenize_korean(self, text: str) -> List[str]:
        """한국어 토큰화 (공백 기반 + 형태소 단위)"""
        if not text or not text.strip():
            return []

        # 기본 전처리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 공백 기반 토큰화
        tokens = text.split()

        # 추가적으로 한글 음절 단위로도 분리 (더 세밀한 매칭을 위해)
        char_tokens = []
        for token in tokens:
            char_tokens.extend(list(token))

        return tokens + char_tokens  # 단어 토큰 + 문자 토큰

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """N-gram 생성"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _calculate_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> Dict[str, float]:
        """ROUGE-N 계산"""
        if not pred_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not pred_ngrams or not ref_ngrams:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)

        # 교집합 계산
        overlap = 0
        for ngram in pred_counter:
            if ngram in ref_counter:
                overlap += min(pred_counter[ngram], ref_counter[ngram])

        # Precision, Recall, F1 계산
        precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0

        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }

    def _calculate_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> Dict[str, float]:
        """ROUGE-L 계산 (Longest Common Subsequence)"""
        if not pred_tokens or not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}

        # LCS 계산
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        lcs_len = lcs_length(pred_tokens, ref_tokens)

        # Precision, Recall, F1 계산
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            fmeasure = 0.0
        else:
            fmeasure = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'fmeasure': fmeasure
        }

    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        pred_tokens = self._tokenize_korean(prediction)
        ref_tokens = self._tokenize_korean(reference)

        # ROUGE-1, ROUGE-2, ROUGE-L 계산
        rouge1 = self._calculate_rouge_n(pred_tokens, ref_tokens, 1)
        rouge2 = self._calculate_rouge_n(pred_tokens, ref_tokens, 2)
        rougeL = self._calculate_rouge_l(pred_tokens, ref_tokens)

        return {
            'rouge1_f1': rouge1['fmeasure'],
            'rouge2_f1': rouge2['fmeasure'],
            'rougeL_f1': rougeL['fmeasure']
        }

    def calculate_max_rouge_scores(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """여러 참조 문장 중 최대 ROUGE 점수 계산"""
        if not references:
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

        max_scores = {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

        for reference in references:
            scores = self.calculate_rouge_scores(prediction, reference)
            max_scores['rouge1_f1'] = max(max_scores['rouge1_f1'], scores['rouge1_f1'])
            max_scores['rouge2_f1'] = max(max_scores['rouge2_f1'], scores['rouge2_f1'])
            max_scores['rougeL_f1'] = max(max_scores['rougeL_f1'], scores['rougeL_f1'])

        return max_scores

    def calculate_final_score(self, prediction: str, references: List[str]) -> float:
        """Final Score 계산"""
        scores = self.calculate_max_rouge_scores(prediction, references)
        return scores['rouge1_f1'] + scores['rouge2_f1'] + scores['rougeL_f1']


# 테스트 함수
if __name__ == "__main__":
    calculator = KoreanRougeCalculator()

    test_cases = [
        ("안녕하세요", "안녕하세요"),  # 완전 일치
        ("안녕하세요 오늘", "안녕하세요 내일"),  # 부분 일치
        ("오늘 날씨가 좋네요", "날씨가 좋네요 오늘"),  # 순서 다름
        ("hello world", "hello world"),  # 영어
        ("", "안녕하세요"),  # 빈 예측
    ]

    print("🔍 한국어 ROUGE 계산기 테스트")
    print("=" * 50)

    for i, (pred, ref) in enumerate(test_cases):
        print(f"\n케이스 {i+1}: '{pred}' vs '{ref}'")

        scores = calculator.calculate_rouge_scores(pred, ref)
        final_score = calculator.calculate_final_score(pred, [ref])

        print(f"  ROUGE-1: {scores['rouge1_f1']:.4f}")
        print(f"  ROUGE-2: {scores['rouge2_f1']:.4f}")
        print(f"  ROUGE-L: {scores['rougeL_f1']:.4f}")
        print(f"  Final Score: {final_score:.4f}")
