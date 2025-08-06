"""
모델 관리 모듈
"""

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from rouge import Rouge
from .custom_metrics import create_compute_metrics_function

class ModelManager:
    """모델 관리 클래스"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, for_training=True):
        """모델과 토크나이저 로드"""
        print('-' * 50)
        print('모델과 토크나이저 로딩 중...')
        print(f'모델명: {self.config["general"]["model_name"]}')
        print(f'디바이스: {self.device}')

        model_name = self.config['general']['model_name']

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if for_training:
            # 학습용 모델 로드
            bart_config = BartConfig.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        else:
            # 추론용 모델 로드 (체크포인트에서)
            ckt_path = self.config['inference']['ckt_path']
            self.model = BartForConditionalGeneration.from_pretrained(ckt_path)

        # Special tokens 추가
        special_tokens_dict = {'additional_special_tokens': self.config['tokenizer']['special_tokens']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # GPU로 이동
        self.model.to(self.device)

        print(f'모델 파라미터 수: {self.model.num_parameters():,}')
        print('모델과 토크나이저 로딩 완료!')
        print('-' * 50)

        return self.model, self.tokenizer

    def compute_metrics(self, pred):
        """
        통합된 Final Score 기반 평가 메트릭 계산 (통합 메트릭 사용)
        Final Score = ROUGE-1 + ROUGE-2 + ROUGE-L
        """
        from utils.unified_metrics import compute_unified_metrics
        predictions, labels = pred
        return compute_unified_metrics(predictions, labels, self.tokenizer, self.config)

    def create_trainer(self, train_dataset, val_dataset):
        """트레이너 생성"""
        print('-' * 50)
        print('트레이너 생성 중...')

        # 학습 인자 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['general']['output_dir'],
            overwrite_output_dir=self.config['training']['overwrite_output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            learning_rate=self.config['training']['learning_rate'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            weight_decay=self.config['training']['weight_decay'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            optim=self.config['training']['optim'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            eval_strategy=self.config['training'].get('eval_strategy', self.config['training'].get('evaluation_strategy', 'epoch')),
            save_strategy=self.config['training']['save_strategy'],
            save_total_limit=self.config['training']['save_total_limit'],
            fp16=self.config['training']['fp16'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            seed=self.config['training']['seed'],
            logging_dir=self.config['training']['logging_dir'],
            logging_strategy=self.config['training']['logging_strategy'],
            predict_with_generate=self.config['training']['predict_with_generate'],
            generation_max_length=self.config['training']['generation_max_length'],
            do_train=self.config['training']['do_train'],
            do_eval=self.config['training']['do_eval'],
            report_to=self.config['training']['report_to']
        )

        # Early stopping 콜백
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config['training']['early_stopping_patience'],
            early_stopping_threshold=self.config['training']['early_stopping_threshold']
        )

        # 트레이너 생성
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )

        print('트레이너 생성 완료!')
        print('-' * 50)

        return trainer

    def optimize_for_gpu(self):
        """GPU 메모리 최적화"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            print("GPU 메모리 최적화 완료")

    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None:
            return "모델이 로드되지 않았습니다."

        info = {
            "model_name": self.config['general']['model_name'],
            "device": str(self.device),
            "parameters": self.model.num_parameters(),
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }

        return info

    def save_model(self, output_dir=None):
        """모델 저장"""
        if output_dir is None:
            output_dir = self.config['general']['output_dir']

        if self.model and self.tokenizer:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"모델이 {output_dir}에 저장되었습니다.")
        else:
            print("저장할 모델이 없습니다.")

class TrainingManager:
    """학습 관리 클래스"""
    def __init__(self, config):
        self.config = config
        self.model_manager = ModelManager(config)

    def train(self, train_dataset, val_dataset):
        """모델 학습 실행"""
        print("=" * 80)
        print("모델 학습 시작")
        print("=" * 80)

        # GPU 최적화
        self.model_manager.optimize_for_gpu()

        # 모델과 토크나이저 로드
        model, tokenizer = self.model_manager.load_model_and_tokenizer(for_training=True)

        # 모델 정보 출력
        info = self.model_manager.get_model_info()
        print(f"모델 정보: {info}")

        # 트레이너 생성
        trainer = self.model_manager.create_trainer(train_dataset, val_dataset)

        # 학습 시작
        print("학습 시작...")
        trainer.train()

        # 모델 저장
        self.model_manager.save_model()

        print("=" * 80)
        print("모델 학습 완료!")
        print("=" * 80)

        return trainer

if __name__ == "__main__":
    # 테스트
    from config_manager import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load_config()

    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    print("모델 매니저 테스트 완료")
