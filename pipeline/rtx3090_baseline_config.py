"""
RTX 3090용 Baseline 설정 (baseline.ipynb 기반)
고성능 GPU에서 baseline.ipynb와 동일한 설정으로 학습
"""

from .config_manager import ConfigManager
from transformers import AutoTokenizer


def create_rtx3090_baseline_config():
    """RTX 3090용 Baseline 설정 생성 (baseline.ipynb 기반)"""
    config_manager = ConfigManager()

    # baseline.ipynb에서 사용하는 모델과 토크나이저
    model_name = "digit82/kobart-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 데이터 경로 자동 감지
    data_path = config_manager.find_latest_augmented_data_path()

    config = {
        "general": {
            "data_path": data_path,
            "model_name": model_name,
            "output_dir": "./model_output/",
            "hardware": "rtx3090"
        },
        "tokenizer": {
            "encoder_max_len": 512,  # baseline.ipynb와 동일
            "decoder_max_len": 100,  # baseline.ipynb와 동일
            "bos_token": f"{tokenizer.bos_token}",
            "eos_token": f"{tokenizer.eos_token}",
            # baseline.ipynb의 special_tokens 사용
            "special_tokens": [
                '#Person1#', '#Person2#', '#Person3#',
                '#PhoneNumber#', '#Address#', '#DateOfBirth#',
                '#PassportNumber#', '#SSN#', '#CardNumber#',
                '#CarNumber#', '#Email#'
            ],
            "padding": "max_length",
            "truncation": True,
            "return_attention_mask": True,
            "return_token_type_ids": False
        },
        "training": {
            # baseline.ipynb의 학습 설정 그대로 적용
            "overwrite_output_dir": True,
            "num_train_epochs": 40,  # baseline.ipynb와 동일
            "learning_rate": 1e-5,   # baseline.ipynb와 동일
            "per_device_train_batch_size": 50,  # baseline.ipynb와 동일 (RTX 3090이므로 가능)
            "per_device_eval_batch_size": 32,   # baseline.ipynb와 동일
            "warmup_ratio": 0.1,     # baseline.ipynb와 동일
            "weight_decay": 0.01,    # baseline.ipynb와 동일
            "lr_scheduler_type": 'cosine',  # baseline.ipynb와 동일
            "optim": 'adamw_torch',  # baseline.ipynb와 동일
            "gradient_accumulation_steps": 1,  # baseline.ipynb와 동일

            # 평가 및 저장 전략 (K-Fold 최적화)
            "evaluation_strategy": 'steps',  # K-Fold에서 더 자주 평가
            "eval_strategy": 'steps',
            "eval_steps": 500,               # 200 스텝마다 평가
            "save_strategy": 'steps',        # K-Fold에서 더 자주 저장
            "save_steps": 500,               # 200 스텝마다 저장
            "save_total_limit": 3,           # K-Fold에서 저장 공간 절약

            # 성능 최적화
            "fp16": True,                    # baseline.ipynb와 동일
            "load_best_model_at_end": True,  # baseline.ipynb와 동일
            "metric_for_best_model": "eval_final_score",
            "greater_is_better": True,

            # 기타 설정
            "seed": 42,                      # baseline.ipynb와 동일
            "logging_dir": "./logs",         # baseline.ipynb와 동일
            "logging_strategy": "steps",     # K-Fold에서 더 자주 로깅
            "logging_steps": 50,             # 더 자주 로깅

            # 생성 관련 설정
            "predict_with_generate": True,   # baseline.ipynb와 동일
            "generation_max_length": 100,    # baseline.ipynb와 동일

            # 학습/평가 활성화
            "do_train": True,                # baseline.ipynb와 동일
            "do_eval": True,                 # baseline.ipynb와 동일

            # Early Stopping (K-Fold 최적화)
            "early_stopping_patience": 5,   # K-Fold에서 더 빠른 조기 종료
            "early_stopping_threshold": 0.001,

            # 로깅 설정
            "report_to": "none",             # baseline.ipynb와 동일

            # RTX 3090 최적화 추가 설정
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "dataloader_drop_last": False,
            "ignore_data_skip": True,

            # 메모리 최적화 (RTX 3090은 24GB VRAM)
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,  # RTX 3090은 메모리가 충분하므로 비활성화

            "output_dir": "./model_output/"
        },
        "wandb": {
            "entity": "your_entity",
            "project": "dialogue_summarization_baseline",
            "name": "rtx3090_baseline_run"
        },
        "inference": {
            "ckt_path": "./model_output/",
            "result_path": "./prediction/",
            "no_repeat_ngram_size": 2,       # baseline.ipynb와 동일
            "early_stopping": True,          # baseline.ipynb와 동일
            "generate_max_length": 100,      # baseline.ipynb와 동일
            "num_beams": 4,                  # baseline.ipynb와 동일
            "batch_size": 32,                # baseline.ipynb와 동일
            # baseline.ipynb의 remove_tokens 사용
            "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
        },
        # LoRA 비활성화 (baseline.ipynb는 Full Fine-tuning)
        "lora": {
            "enabled": False,
            "use_qlora": False
        },
        # K-Fold 활성화 (성능 향상을 위해 기본 활성화)
        "kfold": {
            "enabled": False,
            "n_splits": 5,
            "stratified": True,  # 텍스트 길이 기반 계층화
            "random_state": 42,
            "ensemble_method": "voting",
            "save_individual_models": True,
            "use_ensemble_inference": True
        }
    }

    print("🚀 RTX 3090 Baseline + K-Fold 설정이 생성되었습니다.")
    print("📋 주요 설정:")
    print(f"   - 모델: {model_name}")
    print(f"   - 에포크: {config['training']['num_train_epochs']}")
    print(f"   - 배치 크기: {config['training']['per_device_train_batch_size']}")
    print(f"   - 학습률: {config['training']['learning_rate']}")
    print(f"   - 인코더 길이: {config['tokenizer']['encoder_max_len']}")
    print(f"   - 디코더 길이: {config['tokenizer']['decoder_max_len']}")
    print(f"   - LoRA: 비활성화 (Full Fine-tuning)")
    print(f"   - K-Fold: 활성화 ({config['kfold']['n_splits']} folds, {config['kfold']['ensemble_method']} 앙상블)")
    print(f"   - 계층화: {config['kfold']['stratified']}")

    return config


def create_rtx3090_baseline_kfold_config(n_splits=5, ensemble_method='voting'):
    """RTX 3090용 Baseline + K-Fold 설정"""
    config = create_rtx3090_baseline_config()

    # K-Fold 활성화
    config['kfold'] = {
        'enabled': True,
        'n_splits': n_splits,
        'stratified': True,
        'random_state': 42,
        'ensemble_method': ensemble_method,
        'save_individual_models': True,
        'use_ensemble_inference': True
    }

    # K-Fold에 맞게 학습 설정 조정
    config['training'].update({
        'num_train_epochs': 20,  # fold별로 조금 줄임
        'early_stopping_patience': 2,
        'save_total_limit': 3,
        'evaluation_strategy': 'steps',
        'eval_strategy': 'steps',
        'eval_steps': 200,
        'save_steps': 200,
        'logging_steps': 100
    })

    print(f"🔄 RTX 3090 Baseline + K-Fold 설정이 생성되었습니다.")
    print(f"   - Fold 수: {n_splits}")
    print(f"   - 앙상블 방법: {ensemble_method}")
    print(f"   - 에포크 수: {config['training']['num_train_epochs']} (K-Fold용 조정)")

    return config


def create_rtx3090_baseline_fast_config():
    """RTX 3090용 빠른 Baseline 설정 (실험용)"""
    config = create_rtx3090_baseline_config()

    # 빠른 실험을 위한 설정 조정
    config['training'].update({
        'num_train_epochs': 10,  # 에포크 줄임
        'per_device_train_batch_size': 64,  # 배치 크기 증가 (RTX 3090 활용)
        'per_device_eval_batch_size': 64,
        'early_stopping_patience': 2,
        'evaluation_strategy': 'steps',
        'eval_strategy': 'steps',
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 50
    })

    print("⚡ RTX 3090 빠른 Baseline 설정이 생성되었습니다.")
    print("   - 최적화: 빠른 실험용")
    print(f"   - 에포크: {config['training']['num_train_epochs']}")
    print(f"   - 배치 크기: {config['training']['per_device_train_batch_size']}")

    return config
