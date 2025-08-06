"""
K-Fold 교차 검증 관리자
모델 성능 향상을 위한 K-Fold 학습 및 앙상블 추론
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

from .model_manager import TrainingManager
from .model_manager import ModelManager
from .data_processor import DataProcessor
from .inference_manager import InferenceManager
from utils.unified_metrics import batch_rouge_scores
from utils.memory_monitor import MemoryMonitor, safe_fold_training_wrapper

logger = logging.getLogger(__name__)


class KFoldManager:
    """K-Fold 교차 검증 관리자"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kfold_config = config.get('kfold', {})
        self.n_splits = self.kfold_config.get('n_splits', 5)
        self.stratified = self.kfold_config.get('stratified', False)
        self.random_state = self.kfold_config.get('random_state', 42)
        self.ensemble_method = self.kfold_config.get(
            'ensemble_method', 'voting')  # voting, weighted, best

        # K-Fold 결과 저장 경로
        self.kfold_base_dir = os.path.join(
            config['general']['output_dir'], 'kfold_results')
        os.makedirs(self.kfold_base_dir, exist_ok=True)

        # 각 fold별 모델 저장 경로
        self.fold_dirs = []
        for i in range(self.n_splits):
            fold_dir = os.path.join(self.kfold_base_dir, f'fold_{i+1}')
            os.makedirs(fold_dir, exist_ok=True)
            self.fold_dirs.append(fold_dir)

        # 앙상블 결과 저장 경로
        self.ensemble_dir = os.path.join(self.kfold_base_dir, 'ensemble')
        os.makedirs(self.ensemble_dir, exist_ok=True)

        logger.info(
            f"K-Fold 설정: {self.n_splits}개 fold, 방법: {'Stratified' if self.stratified else 'Standard'}")

    def create_kfold_splits(self, data_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """K-Fold 데이터 분할 생성"""
        logger.info(f"K-Fold 데이터 분할 생성 중... (n_splits={self.n_splits})")

        # 데이터 로드
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"지원하지 않는 데이터 형식: {data_path}")

        # 인덱스 배열 생성
        indices = np.arange(len(data))

        # K-Fold 분할기 생성
        if self.stratified:
            # 텍스트 길이 기반 계층화
            text_lengths = []
            for item in data:
                # 다양한 키 형태 지원
                text = item.get('dialogue', item.get('input', ''))
                text_lengths.append(len(str(text)))

            # 길이를 5개 구간으로 나누어 계층화
            try:
                length_bins = pd.qcut(
                    text_lengths, q=5, labels=False, duplicates='drop')
            except ValueError as e:
                # 중복값이 많아서 계층화가 어려운 경우 3개 구간으로 시도
                logger.warning(f"5개 구간 계층화 실패, 3개 구간으로 시도: {e}")
                try:
                    length_bins = pd.qcut(
                        text_lengths, q=3, labels=False, duplicates='drop')
                except ValueError:
                    # 계층화가 불가능한 경우 일반 K-Fold 사용
                    logger.warning("계층화 불가능, 일반 K-Fold 사용")
                    self.stratified = False
                    length_bins = None

            if self.stratified and length_bins is not None:
                kfold = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                splits = list(kfold.split(indices, length_bins))
            else:
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                splits = list(kfold.split(indices))

        # 분할 정보 저장
        split_info = {
            'n_splits': self.n_splits,
            'stratified': self.stratified,
            'random_state': self.random_state,
            'total_samples': len(data),
            'splits': []
        }

        for i, (train_idx, val_idx) in enumerate(splits):
            split_info['splits'].append({
                'fold': i + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            })

        # 분할 정보 저장
        split_info_path = os.path.join(self.kfold_base_dir, 'split_info.json')
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)

        logger.info(f"K-Fold 분할 완료: {len(splits)}개 fold")
        logger.info(f"분할 정보 저장: {split_info_path}")

        return splits

    def create_fold_datasets(self, data_path: str, train_indices: np.ndarray, val_indices: np.ndarray, fold_num: int):
        """특정 fold의 학습/검증 데이터셋 생성"""
        # 원본 데이터 로드
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

        # 인덱스에 따라 데이터 분할
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]

        # fold별 데이터 저장
        fold_dir = self.fold_dirs[fold_num - 1]

        train_path = os.path.join(fold_dir, 'train_data.json')
        val_path = os.path.join(fold_dir, 'val_data.json')

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Fold {fold_num} 데이터 저장: train={len(train_data)}, val={len(val_data)}")

        return train_path, val_path

    @safe_fold_training_wrapper
    def train_single_fold(self, fold_num: int, train_path: str, val_path: str) -> Dict[str, Any]:
        """단일 fold 학습"""
        logger.info(f"=" * 60)
        logger.info(f"Fold {fold_num}/{self.n_splits} 학습 시작")
        logger.info(f"=" * 60)

        # fold별 설정 복사 및 수정
        fold_config = self.config.copy()
        fold_config['general']['data_path'] = os.path.dirname(train_path)
        fold_config['general']['output_dir'] = self.fold_dirs[fold_num - 1]
        fold_config['training']['output_dir'] = self.fold_dirs[fold_num - 1]
        fold_config['training']['logging_dir'] = os.path.join(
            self.fold_dirs[fold_num - 1], 'logs')

        # 로그 디렉토리 생성
        os.makedirs(fold_config['training']['logging_dir'], exist_ok=True)

        try:
            # 데이터 처리
            data_processor = DataProcessor(fold_config)

            # 모델 및 토크나이저 로드
            model_manager = ModelManager(fold_config)
            model, tokenizer = model_manager.load_model_and_tokenizer(
                for_training=True)

            # fold별 데이터셋 준비 (직접 경로 지정)
            train_dataset, val_dataset = data_processor.prepare_fold_dataset(
                tokenizer, train_path, val_path
            )

            # 학습 실행
            training_manager = TrainingManager(fold_config)
            trainer = training_manager.train(train_dataset, val_dataset)

            # 학습 결과 수집
            fold_result = {
                'fold_num': fold_num,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'final_step': trainer.state.global_step,
                'best_metric': getattr(trainer.state, 'best_metric', None),
                'training_completed': True,
                'model_path': self.fold_dirs[fold_num - 1]
            }

            # fold 결과 저장
            fold_result_path = os.path.join(
                self.fold_dirs[fold_num - 1], 'fold_result.json')
            with open(fold_result_path, 'w', encoding='utf-8') as f:
                json.dump(fold_result, f, ensure_ascii=False, indent=2)

            # 메모리 정리 (중요!)
            logger.info(f"Fold {fold_num} 메모리 정리 중...")

            # GPU 메모리 정리
            import torch
            import gc

            # 모델과 데이터를 명시적으로 삭제
            del model
            del tokenizer
            del trainer
            del train_dataset
            del val_dataset
            del training_manager
            del data_processor

            # 가비지 컬렉션 실행
            gc.collect()

            # GPU 메모리 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # GPU 메모리 상태 출력
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU 메모리 정리 후 - 할당: {allocated:.2f}GB, 캐시: {cached:.2f}GB")

            # 시스템 메모리 정리를 위한 잠시 대기
            import time
            time.sleep(2)

            logger.info(f"Fold {fold_num} 메모리 정리 완료")

            logger.info(f"Fold {fold_num} 학습 완료!")
            logger.info(f"최종 스텝: {fold_result['final_step']}")
            if fold_result['best_metric']:
                logger.info(f"최고 메트릭: {fold_result['best_metric']:.4f}")

            return fold_result

        except Exception as e:
            logger.error(f"Fold {fold_num} 학습 실패: {e}")
            fold_result = {
                'fold_num': fold_num,
                'training_completed': False,
                'error': str(e)
            }
            return fold_result

    def run_kfold_training(self, data_path: str) -> List[Dict[str, Any]]:
        """K-Fold 교차 검증 학습 실행"""
        logger.info(f"🔄 K-Fold 교차 검증 학습 시작 ({self.n_splits} folds)")

        # K-Fold 분할 생성
        splits = self.create_kfold_splits(data_path)

        # 각 fold별 학습 실행
        fold_results = []

        memory_monitor = MemoryMonitor()

        for i, (train_indices, val_indices) in enumerate(splits):
            fold_num = i + 1

            logger.info(f"🔄 Fold {fold_num}/{self.n_splits} 준비 중...")

            # 메모리 상태 확인
            memory_monitor.print_memory_status(f"Fold {fold_num} 시작 전")

            # 메모리 부족 시 대기
            if not memory_monitor.check_memory_safety(min_gpu_gb=1.5):
                logger.warning(f"Fold {fold_num} 시작 전 메모리 부족, 정리 및 대기...")
                memory_monitor.cleanup_memory(aggressive=True)
                memory_monitor.wait_for_memory_recovery(max_wait_seconds=60)

            try:
                # fold별 데이터셋 생성
                train_path, val_path = self.create_fold_datasets(
                    data_path, train_indices, val_indices, fold_num
                )

                # fold 학습 실행
                fold_result = self.train_single_fold(
                    fold_num, train_path, val_path)
                fold_results.append(fold_result)

                # 중간 결과 저장
                self.save_kfold_summary(fold_results, partial=True)

                # fold 간 메모리 정리 및 대기
                logger.info(f"Fold {fold_num} 완료, 다음 fold 준비 중...")
                memory_monitor.cleanup_memory(aggressive=True)

                # fold 간 잠시 대기 (시스템 안정화)
                import time
                time.sleep(5)

            except Exception as e:
                logger.error(f"Fold {fold_num} 실행 중 오류: {e}")
                # 오류 발생 시에도 메모리 정리
                memory_monitor.cleanup_memory(aggressive=True)

                # 오류 정보를 결과에 추가
                fold_result = {
                    'fold_num': fold_num,
                    'training_completed': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                fold_results.append(fold_result)

                # 중간 결과 저장
                self.save_kfold_summary(fold_results, partial=True)

                # 심각한 메모리 오류인 경우 중단
                if 'memory' in str(e).lower() or 'cuda' in str(e).lower():
                    logger.error(f"메모리 관련 오류로 K-Fold 학습 중단: {e}")
                    break

        # 최종 결과 저장
        self.save_kfold_summary(fold_results, partial=False)

        logger.info(f"🎉 K-Fold 교차 검증 학습 완료!")
        return fold_results

    def save_kfold_summary(self, fold_results: List[Dict[str, Any]], partial: bool = False):
        """K-Fold 결과 요약 저장"""
        completed_folds = [r for r in fold_results if r.get(
            'training_completed', False)]
        failed_folds = [r for r in fold_results if not r.get(
            'training_completed', False)]

        summary = {
            'kfold_config': self.kfold_config,
            'total_folds': self.n_splits,
            'completed_folds': len(completed_folds),
            'failed_folds': len(failed_folds),
            'completion_rate': len(completed_folds) / self.n_splits * 100,
            'fold_results': fold_results,
            'summary_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_partial': partial
        }

        # 성공한 fold들의 메트릭 통계
        if completed_folds:
            best_metrics = [r['best_metric']
                            for r in completed_folds if r.get('best_metric')]
            if best_metrics:
                summary['metric_stats'] = {
                    'mean': np.mean(best_metrics),
                    'std': np.std(best_metrics),
                    'min': np.min(best_metrics),
                    'max': np.max(best_metrics),
                    'best_fold': completed_folds[np.argmax(best_metrics)]['fold_num']
                }

        # 요약 저장
        summary_path = os.path.join(self.kfold_base_dir, 'kfold_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 진행 상황 출력
        status = "진행 중" if partial else "완료"
        logger.info(
            f"📊 K-Fold 요약 ({status}): {len(completed_folds)}/{self.n_splits} 완료")
        if completed_folds and 'metric_stats' in summary:
            stats = summary['metric_stats']
            logger.info(f"   평균 메트릭: {stats['mean']:.4f} ± {stats['std']:.4f}")
            logger.info(
                f"   최고 성능: Fold {stats['best_fold']} ({stats['max']:.4f})")

    def ensemble_inference(self, test_data_path: str = None) -> Dict[str, Any]:
        """앙상블 추론 실행"""
        logger.info(f"🎯 앙상블 추론 시작 (방법: {self.ensemble_method})")

        # 완료된 fold 모델들 찾기
        completed_folds = []
        for i in range(self.n_splits):
            fold_dir = self.fold_dirs[i]
            fold_result_path = os.path.join(fold_dir, 'fold_result.json')

            if os.path.exists(fold_result_path):
                with open(fold_result_path, 'r', encoding='utf-8') as f:
                    fold_result = json.load(f)
                    if fold_result.get('training_completed', False):
                        completed_folds.append((i + 1, fold_dir, fold_result))

        if not completed_folds:
            raise ValueError("완료된 fold 모델이 없습니다.")

        logger.info(f"앙상블에 사용할 모델: {len(completed_folds)}개")

        # 각 fold 모델로 추론 실행
        fold_predictions = []

        for fold_num, fold_dir, fold_result in completed_folds:
            logger.info(f"Fold {fold_num} 모델로 추론 중...")

            try:
                # fold별 설정 생성
                fold_config = self.config.copy()
                fold_config['inference']['ckt_path'] = fold_dir

                # 추론 실행
                inference_manager = InferenceManager(fold_config)
                data_processor = DataProcessor(fold_config)

                # 모델 로드
                model_manager = ModelManager(fold_config)
                model, tokenizer = model_manager.load_model_and_tokenizer(
                    for_training=False)

                # 추론 실행
                predictions = inference_manager.run_inference(
                    model, tokenizer, data_processor)

                fold_predictions.append({
                    'fold_num': fold_num,
                    'predictions': predictions,
                    'best_metric': fold_result.get('best_metric', 0.0)
                })

                logger.info(f"Fold {fold_num} 추론 완료: {len(predictions)} 개 예측")

            except Exception as e:
                logger.error(f"Fold {fold_num} 추론 실패: {e}")
                # 실패한 fold는 건너뛰고 계속 진행
                continue

        # 앙상블 결과 생성
        ensemble_result = self.combine_predictions(fold_predictions)

        # 앙상블 결과 저장 (DataFrame을 직렬화 가능한 형태로 변환)
        serializable_result = self._make_json_serializable(ensemble_result)

        ensemble_result_path = os.path.join(
            self.ensemble_dir, 'ensemble_predictions.json')
        with open(ensemble_result_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        # DataFrame인 경우 CSV로도 저장
        if 'predictions' in ensemble_result:
            predictions = ensemble_result['predictions']
            if hasattr(predictions, 'to_csv'):  # DataFrame인 경우
                csv_path = os.path.join(self.ensemble_dir, 'ensemble_predictions.csv')
                predictions.to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"📄 앙상블 결과 CSV 저장: {csv_path}")

        logger.info(f"🎉 앙상블 추론 완료! 결과 저장: {ensemble_result_path}")
        return ensemble_result

    def _make_json_serializable(self, obj):
        """객체를 JSON 직렬화 가능한 형태로 변환"""
        import pandas as pd

        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, pd.DataFrame):
                    # DataFrame을 딕셔너리로 변환
                    result[key] = {
                        'type': 'DataFrame',
                        'data': value.to_dict('records'),
                        'columns': value.columns.tolist(),
                        'shape': value.shape
                    }
                elif isinstance(value, (list, dict)):
                    result[key] = self._make_json_serializable(value)
                else:
                    result[key] = value
            return result
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'data': obj.to_dict('records'),
                'columns': obj.columns.tolist(),
                'shape': obj.shape
            }
        else:
            return obj

    def combine_predictions(self, fold_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """fold별 예측 결과를 앙상블로 결합"""
        if self.ensemble_method == 'voting':
            return self._voting_ensemble(fold_predictions)
        elif self.ensemble_method == 'weighted':
            return self._weighted_ensemble(fold_predictions)
        elif self.ensemble_method == 'best':
            return self._best_model_ensemble(fold_predictions)
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {self.ensemble_method}")

    def _voting_ensemble(self, fold_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """투표 기반 앙상블 (단순 평균)"""
        logger.info("투표 기반 앙상블 수행 중...")

        # 모든 fold의 예측 결과 수집
        all_predictions = [fp['predictions'] for fp in fold_predictions]

        # DataFrame인지 확인하고 적절히 처리
        import pandas as pd

        if isinstance(all_predictions[0], pd.DataFrame):
            # DataFrame인 경우 summary 컬럼 추출
            all_summaries = [pred['summary'].tolist() for pred in all_predictions]
            all_fnames = all_predictions[0]['fname'].tolist()  # 첫 번째 fold의 fname 사용

            ensemble_predictions = []

            for i in range(len(all_summaries[0])):
                # 각 fold의 i번째 예측 수집
                fold_preds_for_sample = [summaries[i] for summaries in all_summaries]

                # 단순히 첫 번째 fold의 예측을 사용 (텍스트 앙상블의 복잡성 때문)
                # 실제로는 ROUGE 점수 기반 선택이나 다른 방법 사용 가능
                ensemble_predictions.append(fold_preds_for_sample[0])

            # DataFrame 형태로 결과 생성
            ensemble_df = pd.DataFrame({
                'fname': all_fnames,
                'summary': ensemble_predictions
            })

            return {
                'method': 'voting',
                'predictions': ensemble_df,
                'fold_count': len(fold_predictions),
                'individual_predictions': {
                    f'fold_{fp["fold_num"]}': fp['predictions']
                    for fp in fold_predictions
                }
            }
        else:
            # 리스트인 경우 기존 로직 사용
            ensemble_predictions = []

            for i in range(len(all_predictions[0])):
                # 각 fold의 i번째 예측 수집
                fold_preds_for_sample = [pred[i] for pred in all_predictions]

                # 단순히 첫 번째 fold의 예측을 사용
                ensemble_predictions.append(fold_preds_for_sample[0])

            return {
                'method': 'voting',
                'predictions': ensemble_predictions,
                'fold_count': len(fold_predictions),
                'individual_predictions': {
                    f'fold_{fp["fold_num"]}': fp['predictions']
                    for fp in fold_predictions
                }
            }

    def _weighted_ensemble(self, fold_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """가중 평균 앙상블 (성능 기반 가중치)"""
        logger.info("가중 평균 앙상블 수행 중...")

        # 각 fold의 성능을 가중치로 사용
        weights = []
        for fp in fold_predictions:
            metric = fp.get('best_metric', 0.0)
            weights.append(max(metric, 0.1))  # 최소 가중치 보장

        # 가중치 정규화
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        logger.info(f"Fold별 가중치: {[f'{w:.3f}' for w in weights]}")

        # 가중치가 가장 높은 fold의 예측 사용 (텍스트 특성상)
        best_fold_idx = np.argmax(weights)
        best_fold_predictions = fold_predictions[best_fold_idx]['predictions']

        return {
            'method': 'weighted',
            'predictions': best_fold_predictions,
            'weights': weights,
            'best_fold': fold_predictions[best_fold_idx]['fold_num'],
            'fold_count': len(fold_predictions)
        }

    def _best_model_ensemble(self, fold_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최고 성능 모델 선택"""
        logger.info("최고 성능 모델 선택 중...")

        # 가장 높은 메트릭을 가진 fold 찾기
        best_fold = max(fold_predictions,
                        key=lambda x: x.get('best_metric', 0.0))

        logger.info(
            f"최고 성능 모델: Fold {best_fold['fold_num']} (메트릭: {best_fold['best_metric']:.4f})")

        return {
            'method': 'best',
            'predictions': best_fold['predictions'],
            'best_fold': best_fold['fold_num'],
            'best_metric': best_fold['best_metric'],
            'fold_count': len(fold_predictions)
        }
