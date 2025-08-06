"""
설정 관리 모듈 - 모델별, 하드웨어별 최적화 설정 관리
"""

import yaml
import os
import glob
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class HardwareSpec:
    """하드웨어 사양 정의"""
    name: str
    vram_gb: float
    compute_capability: float
    memory_bandwidth: int
    tensor_cores: bool = False


@dataclass
class ModelSpec:
    """모델 사양 정의"""
    name: str
    model_id: str
    size: str  # 'small', 'base', 'large'
    parameters: int  # 파라미터 수 (millions)
    recommended_vram: float
    supports_korean: bool = False
    supports_lora: bool = True  # LoRA 지원 여부


@dataclass
class LoRAConfig:
    """LoRA 설정 정의"""
    enabled: bool = False
    use_qlora: bool = False  # QLoRA 사용 여부
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha
    dropout: float = 0.1  # LoRA dropout
    target_modules: list = None  # 타겟 모듈들
    bias: str = "none"  # bias 설정
    task_type: str = "SEQ_2_SEQ_LM"  # 태스크 타입


class ConfigPresets:
    """설정 프리셋 관리 클래스"""

    # 하드웨어 사양 정의
    HARDWARE_SPECS = {
        'rtx3060': HardwareSpec('RTX 3060', 6.0, 8.6, 360, True),
        'rtx3070': HardwareSpec('RTX 3070', 8.0, 8.6, 448, True),
        'rtx3080': HardwareSpec('RTX 3080', 10.0, 8.6, 760, True),
        'rtx3090': HardwareSpec('RTX 3090', 24.0, 8.6, 936, True),
        'rtx4070': HardwareSpec('RTX 4070', 12.0, 8.9, 504, True),
        'rtx4080': HardwareSpec('RTX 4080', 16.0, 8.9, 717, True),
        'rtx4090': HardwareSpec('RTX 4090', 24.0, 8.9, 1008, True),
        'cpu': HardwareSpec('CPU', 0.0, 0.0, 0, False),
    }

    # 모델 사양 정의
    MODEL_SPECS = {
        'kobart-base-v2': ModelSpec(
            'KoBART Base v2', 'gogamza/kobart-base-v2', 'base', 124, 4.0, True, True
        ),
        'kobart-summarization-large': ModelSpec(
            'KoBART Summarization', 'gogamza/kobart-summarization', 'large', 124, 6.0, True, True
        ),
        'bart-base': ModelSpec(
            'BART Base', 'facebook/bart-base', 'base', 139, 4.0, False, True
        ),
        'bart-large': ModelSpec(
            'BART Large', 'facebook/bart-large', 'large', 406, 8.0, False, True
        ),
        'kobart-summarization': ModelSpec(
            'KoBART Summarization', 'digit82/kobart-summarization', 'base', 124, 4.0, True, True
        ),
        # 큰 모델들 (LoRA/QLoRA 전용)
        'bart-large-lora': ModelSpec(
            'BART Large (LoRA)', 'facebook/bart-large', 'large', 406, 4.0, False, True
        ),
        'kobart-large-lora': ModelSpec(
            'KoBART Large (LoRA)', 'gogamza/kobart-summarization', 'large', 124, 3.0, True, True
        ),
    }

    # LoRA 설정 프리셋
    LORA_PRESETS = {
        'disabled': LoRAConfig(enabled=False),
        'lora_light': LoRAConfig(
            enabled=True, use_qlora=False, r=8, alpha=16, dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        ),
        'lora_standard': LoRAConfig(
            enabled=True, use_qlora=False, r=16, alpha=32, dropout=0.1,
            target_modules=["q_proj", "v_proj",
                            "k_proj", "out_proj", "fc1", "fc2"]
        ),
        'lora_heavy': LoRAConfig(
            enabled=True, use_qlora=False, r=32, alpha=64, dropout=0.1,
            target_modules=["q_proj", "v_proj",
                            "k_proj", "out_proj", "fc1", "fc2"]
        ),
        'qlora_standard': LoRAConfig(
            enabled=True, use_qlora=True, r=16, alpha=32, dropout=0.1,
            target_modules=["q_proj", "v_proj",
                            "k_proj", "out_proj", "fc1", "fc2"]
        ),
        'qlora_heavy': LoRAConfig(
            enabled=True, use_qlora=True, r=32, alpha=64, dropout=0.1,
            target_modules=["q_proj", "v_proj",
                            "k_proj", "out_proj", "fc1", "fc2"]
        ),
    }

    @staticmethod
    def get_optimal_model_for_hardware(hardware_key: str) -> str:
        """하드웨어에 최적화된 모델 선택"""
        hardware = ConfigPresets.HARDWARE_SPECS.get(hardware_key)
        if not hardware:
            return 'kobart-base-v2'

        # VRAM 기반 모델 선택
        if hardware.vram_gb >= 12.0:
            return 'kobart-summarization-large'  # 큰 모델 사용 가능
        elif hardware.vram_gb >= 6.0:
            return 'kobart-base-v2'   # 중간 모델
        else:
            return 'kobart-base-v2'   # 안전한 선택

    @staticmethod
    def get_optimal_lora_config(hardware_key: str, model_key: str) -> str:
        """하드웨어와 모델에 최적화된 LoRA 설정 선택"""
        hardware = ConfigPresets.HARDWARE_SPECS.get(hardware_key)
        model = ConfigPresets.MODEL_SPECS.get(model_key)

        if not hardware or not model:
            return 'disabled'

        # 큰 모델이거나 VRAM이 부족한 경우 LoRA 사용
        if model.size == 'large' and hardware.vram_gb < 16.0:
            if hardware.vram_gb >= 8.0:
                return 'qlora_standard'  # QLoRA로 메모리 절약
            else:
                return 'qlora_heavy'     # 더 강한 QLoRA
        elif model.size == 'large' and hardware.vram_gb >= 16.0:
            return 'lora_standard'       # 충분한 메모리에서는 일반 LoRA
        elif hardware.vram_gb < 6.0:
            return 'lora_light'          # 저사양에서는 가벼운 LoRA
        else:
            return 'disabled'            # 일반 학습

    @staticmethod
    def get_training_config(hardware_key: str, model_key: str) -> Dict[str, Any]:
        """하드웨어와 모델에 최적화된 학습 설정"""
        hardware = ConfigPresets.HARDWARE_SPECS.get(hardware_key)
        model = ConfigPresets.MODEL_SPECS.get(model_key)

        if not hardware or not model:
            return ConfigPresets._get_default_training_config()

        # 기본 설정
        config = {
            "num_train_epochs": 6,
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "fp16": hardware.tensor_cores,
            "gradient_checkpointing": hardware.vram_gb < 12.0,
            "dataloader_pin_memory": hardware.vram_gb > 0,
            "save_total_limit": 3,
        }

        # 하드웨어별 최적화
        if hardware.vram_gb >= 20.0:  # RTX 3090, 4090
            config.update({
                "per_device_train_batch_size": 16 if model.size == 'base' else 12,
                "per_device_eval_batch_size": 16 if model.size == 'base' else 12,
                "gradient_accumulation_steps": 1,
                "dataloader_num_workers": 6,
                "eval_steps": 150,
                "save_steps": 150,
                "num_train_epochs": 8,
                "save_total_limit": 5,
            })
        elif hardware.vram_gb >= 12.0:  # RTX 4070, 3080
            config.update({
                "per_device_train_batch_size": 12 if model.size == 'base' else 8,
                "per_device_eval_batch_size": 12 if model.size == 'base' else 8,
                "gradient_accumulation_steps": 2,
                "dataloader_num_workers": 4,
                "eval_steps": 200,
                "save_steps": 200,
            })
        elif hardware.vram_gb >= 6.0:  # RTX 3060, 3070 - Final Score 최적화
            config.update({
                # 배치 크기 조정 (안정성)
                "per_device_train_batch_size": 12 if model.size == 'base' else 6,
                "per_device_eval_batch_size": 16 if model.size == 'base' else 8,   # 평가는 더 큰 배치
                "gradient_accumulation_steps": 3,  # 효과적인 배치 크기 = 12*3=36
                "dataloader_num_workers": 4,
                "eval_steps": 300,  # 더 자주 평가하여 최적 모델 찾기
                "save_steps": 300,
                "num_train_epochs": 5,  # 에포크 증가로 더 나은 수렴
                "gradient_checkpointing": True,  # 메모리 효율성으로 더 큰 모델 학습
                "learning_rate": 2e-5,  # 낮은 학습률로 안정적 학습
                "warmup_ratio": 0.15,  # 더 긴 워밍업으로 안정적 시작
                "weight_decay": 0.02,  # 정규화 강화
                "lr_scheduler_type": "cosine_with_restarts",  # 더 나은 스케줄러
                "cosine_restarts": 2,  # 재시작으로 local minima 탈출
            })
        else:  # CPU
            config.update({
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 16,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": False,
                "fp16": False,
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "num_train_epochs": 3,
            })

        # 모델별 학습률 조정
        if model.size == 'large':
            config["learning_rate"] = 2e-5  # 큰 모델은 낮은 학습률
        elif model.size == 'small':
            config["learning_rate"] = 5e-5  # 작은 모델은 높은 학습률

        return config

    @staticmethod
    def get_tokenizer_config(hardware_key: str, model_key: str) -> Dict[str, Any]:
        """하드웨어와 모델에 최적화된 토크나이저 설정"""
        hardware = ConfigPresets.HARDWARE_SPECS.get(hardware_key)
        model = ConfigPresets.MODEL_SPECS.get(model_key)

        # 기본 설정
        config = {
            "encoder_max_len": 256,
            "decoder_max_len": 64,
        }

        # 하드웨어별 시퀀스 길이 조정 (RTX 3060 최적화)
        if hardware and hardware.vram_gb >= 20.0:
            config.update({
                "encoder_max_len": 768 if model and model.size == 'large' else 512,
                "decoder_max_len": 192 if model and model.size == 'large' else 128,
            })
        elif hardware and hardware.vram_gb >= 12.0:
            config.update({
                "encoder_max_len": 512,
                "decoder_max_len": 128,
            })
        elif hardware and hardware.vram_gb >= 6.0:  # RTX 3060 Final Score 최적화
            config.update({
                "encoder_max_len": 384,  # 더 긴 컨텍스트로 정보 보존
                "decoder_max_len": 96,   # 더 긴 출력으로 품질 향상
            })
        elif hardware and hardware.vram_gb < 6.0:  # CPU
            config.update({
                "encoder_max_len": 128,
                "decoder_max_len": 32,
            })

        return config

    @staticmethod
    def get_inference_config(hardware_key: str, model_key: str) -> Dict[str, Any]:
        """하드웨어와 모델에 최적화된 인퍼런스 설정"""
        hardware = ConfigPresets.HARDWARE_SPECS.get(hardware_key)

        # 기본 설정
        config = {
            "batch_size": 32,
            "num_beams": 4,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "length_penalty": 1.2,
            "repetition_penalty": 1.1,
        }

        # 하드웨어별 배치 크기 조정
        if hardware and hardware.vram_gb >= 20.0:
            config.update({
                "batch_size": 96,
                "num_beams": 6,
                "no_repeat_ngram_size": 4,
                "length_penalty": 1.3,
                "repetition_penalty": 1.15,
            })
        elif hardware and hardware.vram_gb >= 12.0:
            config.update({
                "batch_size": 64,
                "num_beams": 5,
                "no_repeat_ngram_size": 3,
            })
        elif hardware and hardware.vram_gb < 6.0:  # CPU
            config.update({
                "batch_size": 4,
                "num_beams": 3,
            })

        return config

    @staticmethod
    def _get_default_training_config() -> Dict[str, Any]:
        """기본 학습 설정"""
        return {
            "num_train_epochs": 6,
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": 300,
            "save_steps": 300,
            "fp16": True,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 2,
            "dataloader_pin_memory": True,
            "save_total_limit": 3,
        }


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = None
        self.presets = ConfigPresets()

    @staticmethod
    def find_latest_augmented_data_path(base_path: str = "./data") -> str:
        """최신 증강 데이터 폴더 자동 탐지"""
        try:
            # augmented_로 시작하는 폴더들 찾기
            pattern = os.path.join(base_path, "augmented_*")
            augmented_folders = glob.glob(pattern)

            if not augmented_folders:
                print(f"⚠️ {base_path}에서 증강 데이터 폴더를 찾을 수 없습니다.")
                return base_path

            # 폴더명에서 타임스탬프 추출하여 정렬
            def extract_timestamp(folder_path):
                folder_name = os.path.basename(folder_path)
                # augmented_method_path_ratio_timestamp 형식에서 timestamp 추출
                parts = folder_name.split('_')
                if len(parts) >= 5:
                    # 마지막 두 부분이 날짜와 시간 (YYYYMMDD_HHMMSS)
                    try:
                        timestamp = f"{parts[-2]}_{parts[-1]}"
                        return timestamp
                    except:
                        pass
                return "00000000_000000"  # 기본값

            # 타임스탬프 기준으로 내림차순 정렬 (최신 순)
            augmented_folders.sort(key=extract_timestamp, reverse=True)
            latest_folder = augmented_folders[0]

            print(f"🔍 최신 증강 데이터 폴더 감지: {os.path.basename(latest_folder)}")
            return latest_folder

        except Exception as e:
            print(f"⚠️ 증강 데이터 폴더 탐지 실패: {e}")
            return base_path

    def create_optimized_config(self,
                                hardware_key: Optional[str] = None,
                                model_key: Optional[str] = None,
                                data_path: Optional[str] = None) -> Dict[str, Any]:
        """하드웨어와 모델에 최적화된 설정 생성"""

        # 데이터 경로 자동 감지
        if data_path is None:
            data_path = self.find_latest_augmented_data_path()

        # 하드웨어 자동 감지
        if hardware_key is None:
            hardware_key = self._detect_hardware()

        # 모델 자동 선택
        if model_key is None:
            model_key = self.presets.get_optimal_model_for_hardware(
                hardware_key)

        # 모델 정보 가져오기
        model_spec = self.presets.MODEL_SPECS.get(model_key)
        if not model_spec:
            raise ValueError(f"Unknown model key: {model_key}")

        # 토크나이저 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id)
        except Exception as e:
            print(f"⚠️ 토크나이저 로드 실패: {e}")
            tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

        # LoRA 설정 결정
        lora_preset_key = self.presets.get_optimal_lora_config(
            hardware_key, model_key)
        lora_config = self.presets.LORA_PRESETS.get(lora_preset_key)

        # 설정 구성
        config_data = {
            "general": {
                "data_path": data_path,
                "model_name": model_spec.model_id,
                "output_dir": "./model_output/",
                "hardware": hardware_key,
                "model_key": model_key,
                "lora_preset": lora_preset_key,
            },
            "lora": {
                "enabled": lora_config.enabled,
                "use_qlora": lora_config.use_qlora,
                "r": lora_config.r,
                "alpha": lora_config.alpha,
                "dropout": lora_config.dropout,
                "target_modules": lora_config.target_modules or ["q_proj", "v_proj"],
                "bias": lora_config.bias,
                "task_type": lora_config.task_type,
            },
            "tokenizer": {
                **self.presets.get_tokenizer_config(hardware_key, model_key),
                "bos_token": str(tokenizer.bos_token),
                "eos_token": str(tokenizer.eos_token),
                "special_tokens": [
                    'A:', 'B:',
                    '#PhoneNumber#', '#Address#', '#DateOfBirth#',
                    '#PassportNumber#', '#SSN#', '#CardNumber#',
                    '#CarNumber#', '#Email#'
                ]
            },
            "training": {
                **self.presets.get_training_config(hardware_key, model_key),
                "overwrite_output_dir": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_final_score",
                "greater_is_better": True,
                "seed": 42,
                "logging_dir": "./logs",
                "logging_strategy": "steps",
                "logging_steps": 50,
                "predict_with_generate": True,
                "generation_max_length": self.presets.get_tokenizer_config(hardware_key, model_key)["decoder_max_len"],
                "do_train": True,
                "do_eval": True,
                "early_stopping_patience": 3,
                "early_stopping_threshold": 0.001,
                "report_to": "none",
                "lr_scheduler_type": "cosine",
                "optim": "adamw_torch",
            },
            "inference": {
                **self.presets.get_inference_config(hardware_key, model_key),
                "ckt_path": "./model_output/",
                "result_path": "./prediction/",
                "generate_max_length": self.presets.get_tokenizer_config(hardware_key, model_key)["decoder_max_len"],
                "remove_tokens": ['<usr>', str(tokenizer.bos_token), str(tokenizer.eos_token), str(tokenizer.pad_token)]
            },
            "wandb": {
                "entity": "your_entity",
                "project": "dialogue_summarization",
                "name": f"{model_key}_{hardware_key}_run"
            }
        }

        return config_data

    def _detect_hardware(self) -> str:
        """하드웨어 자동 감지"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                gpu_memory = torch.cuda.get_device_properties(
                    0).total_memory / (1024**3)

                print(f"🖥️ 감지된 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU 메모리: {gpu_memory:.1f}GB")

                # GPU 이름 기반 매칭
                for key in self.presets.HARDWARE_SPECS.keys():
                    if key.replace('rtx', 'rtx ') in gpu_name:
                        return key

                # 메모리 기반 매칭
                if gpu_memory >= 22:
                    return 'rtx4090'
                elif gpu_memory >= 20:
                    return 'rtx3090'
                elif gpu_memory >= 14:
                    return 'rtx4080'
                elif gpu_memory >= 10:
                    return 'rtx4070'
                elif gpu_memory >= 8:
                    return 'rtx3070'
                else:
                    return 'rtx3060'
            else:
                return 'cpu'
        except Exception as e:
            print(f"⚠️ 하드웨어 감지 실패: {e}")
            return 'rtx3060'  # 기본값

    def create_default_config(self, model_name="digit82/kobart-summarization", data_path=None):
        """기본 설정 생성"""
        # 데이터 경로 자동 감지
        if data_path is None:
            data_path = self.find_latest_augmented_data_path()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        config_data = {
            "general": {
                "data_path": data_path,
                "model_name": model_name,
                "output_dir": "./model_output/"
            },
            "tokenizer": {
                "encoder_max_len": 512,
                "decoder_max_len": 100,
                "bos_token": f"{tokenizer.bos_token}",
                "eos_token": f"{tokenizer.eos_token}",
                "special_tokens": [
                    'A:', 'B:',
                    '#PhoneNumber#', '#Address#', '#DateOfBirth#',
                    '#PassportNumber#', '#SSN#', '#CardNumber#',
                    '#CarNumber#', '#Email#'
                ]
            },
            "training": {
                "overwrite_output_dir": True,
                "num_train_epochs": 10,
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 16,
                "per_device_eval_batch_size": 16,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "lr_scheduler_type": 'cosine',
                "optim": 'adamw_torch',
                "gradient_accumulation_steps": 2,
                "evaluation_strategy": 'epoch',
                "save_strategy": 'epoch',
                "save_total_limit": 3,
                "fp16": True,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_final_score",
                "greater_is_better": True,
                "seed": 42,
                "logging_dir": "./logs",
                "logging_strategy": "epoch",
                "predict_with_generate": True,
                "generation_max_length": 100,
                "do_train": True,
                "do_eval": True,
                "early_stopping_patience": 3,
                "early_stopping_threshold": 0.001,
                "report_to": "none"
            },
            "wandb": {
                "entity": "your_entity",
                "project": "dialogue_summarization",
                "name": "baseline_run"
            },
            "inference": {
                "ckt_path": "./model_output/",
                "result_path": "./prediction/",
                "no_repeat_ngram_size": 2,
                "early_stopping": True,
                "generate_max_length": 100,
                "num_beams": 4,
                "batch_size": 32,
                "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
            },
            "kfold": {
                "enabled": False,
                "n_splits": 5,
                "stratified": False,
                "random_state": 42,
                "ensemble_method": "voting",  # voting, weighted, best
                "save_individual_models": True,
                "use_ensemble_inference": True
            }
        }

        return config_data

    def save_config(self, config_data):
        """설정을 YAML 파일로 저장"""
        with open(self.config_path, "w", encoding='utf-8') as file:
            yaml.dump(config_data, file, allow_unicode=True,
                      default_flow_style=False)
        print(f"설정이 {self.config_path}에 저장되었습니다.")

    def load_config(self):
        """YAML 파일에서 설정 로드"""
        if not os.path.exists(self.config_path):
            print(f"{self.config_path}가 없습니다. 기본 설정을 생성합니다.")
            config_data = self.create_default_config()
            self.save_config(config_data)

        with open(self.config_path, "r", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        return self.config

    def update_config(self, updates):
        """설정 업데이트"""
        if self.config is None:
            self.load_config()

        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.config, updates)
        self.save_config(self.config)

        return self.config

    def get_config(self):
        """현재 설정 반환"""
        if self.config is None:
            self.load_config()
        return self.config

    def print_config(self):
        """설정 출력"""
        if self.config is None:
            self.load_config()

        print("=== 현재 설정 ===")
        for section, values in self.config.items():
            print(f"\n[{section}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")

# 편의 함수들 (하위 호환성)


def create_optimized_config_for_rtx3060():
    """RTX 3060에 최적화된 설정 생성 - Final Score 향상 버전"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # Final Score 향상을 위한 최적화
    config['training'].update({
        # 학습 안정성 향상
        "learning_rate": 2e-5,              # 낮은 학습률로 안정적 수렴
        "per_device_train_batch_size": 12,  # 적절한 배치 크기
        "gradient_accumulation_steps": 3,   # 효과적 배치 = 36
        "weight_decay": 0.02,               # 강화된 정규화
        "max_grad_norm": 0.5,               # 강한 그래디언트 클리핑
        "warmup_ratio": 0.15,               # 긴 워밍업

        # 학습 일정 최적화
        "num_train_epochs": 5,              # 충분한 학습
        "eval_steps": 300,                  # 자주 평가
        "save_steps": 300,
        "early_stopping_patience": 5,      # 긴 patience
        "early_stopping_threshold": 0.005,  # 민감한 threshold

        # 생성 품질 향상
        "generation_num_beams": 5,          # 더 많은 beam
        "generation_length_penalty": 1.1,  # 적절한 penalty
        "generation_no_repeat_ngram_size": 3,  # 반복 방지 강화

        # 최적화기 설정
        "adam_epsilon": 1e-6,               # 안정성 향상
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,                 # 더 나은 수렴

        # 기타 최적화
        "dataloader_drop_last": False,
        "logging_steps": 25,
        "save_total_limit": 5,
        "lr_scheduler_type": "cosine_with_restarts",
        "cosine_restarts": 2,
    })

    # 토크나이저 품질 향상
    config['tokenizer'].update({
        "encoder_max_len": 384,             # 긴 컨텍스트
        "decoder_max_len": 96,              # 긴 출력
        "padding": "max_length",
        "truncation": True,
        "return_attention_mask": True,
    })

    # 추론 품질 향상
    config['inference'].update({
        "num_beams": 5,
        "length_penalty": 1.1,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "do_sample": False,
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_high_quality_config_for_rtx3060():
    """RTX 3060용 고품질 Final Score 최적화 설정"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # 고품질 학습을 위한 설정
    config['training'].update({
        # 학습 파라미터 최적화
        'num_train_epochs': 8,              # 충분한 학습
        'learning_rate': 1.5e-5,            # 매우 낮은 학습률
        'per_device_train_batch_size': 8,   # 작은 배치로 안정성
        'per_device_eval_batch_size': 12,
        'gradient_accumulation_steps': 6,   # 효과적 배치 = 48

        # 스케줄링 최적화
        'warmup_ratio': 0.2,                # 긴 워밍업
        'lr_scheduler_type': 'polynomial',  # 부드러운 감소
        'polynomial_decay_power': 2.0,

        # 정규화 강화
        'weight_decay': 0.05,               # 강한 정규화
        'max_grad_norm': 0.3,               # 강한 클리핑
        'label_smoothing_factor': 0.1,      # 라벨 스무딩

        # 평가 및 저장 최적화
        'eval_steps': 200,                  # 자주 평가
        'save_steps': 200,
        'early_stopping_patience': 8,      # 매우 긴 patience
        'early_stopping_threshold': 0.001,

        # 생성 품질 최대화
        'generation_num_beams': 8,          # 최대 beam
        'generation_length_penalty': 1.2,
        'generation_no_repeat_ngram_size': 4,
        'generation_do_sample': False,

        # 메모리 최적화
        'gradient_checkpointing': True,
        'dataloader_pin_memory': True,
        'dataloader_num_workers': 2,

        # 기타 최적화
        'fp16': True,
        'dataloader_drop_last': False,
        'ignore_data_skip': True,
    })

    # 토크나이저 고품질 설정
    config['tokenizer'].update({
        'encoder_max_len': 512,             # 긴 컨텍스트
        'decoder_max_len': 128,             # 긴 출력
        'padding': 'max_length',
        'truncation': True,
        'return_attention_mask': True,
    })

    # 추론 고품질 설정
    config['inference'].update({
        'num_beams': 8,
        'length_penalty': 1.2,
        'repetition_penalty': 1.02,
        'no_repeat_ngram_size': 4,
        'early_stopping': True,
        'do_sample': False,
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_balanced_config_for_rtx3060():
    """RTX 3060용 속도와 품질의 균형 잡힌 설정"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # 균형 잡힌 설정
    config['training'].update({
        'num_train_epochs': 6,
        'learning_rate': 2.5e-5,
        'per_device_train_batch_size': 10,
        'per_device_eval_batch_size': 14,
        'gradient_accumulation_steps': 4,   # 효과적 배치 = 40

        'warmup_ratio': 0.12,
        'weight_decay': 0.03,
        'max_grad_norm': 0.4,

        'eval_steps': 250,
        'save_steps': 250,
        'early_stopping_patience': 6,

        'generation_num_beams': 6,
        'generation_length_penalty': 1.15,
        'generation_no_repeat_ngram_size': 3,

        'gradient_checkpointing': True,
        'dataloader_num_workers': 3,
    })

    config['tokenizer'].update({
        'encoder_max_len': 384,
        'decoder_max_len': 96,
    })

    config['inference'].update({
        'num_beams': 6,
        'length_penalty': 1.15,
        'repetition_penalty': 1.03,
        'no_repeat_ngram_size': 3,
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_optimized_config_for_rtx3090():
    """RTX 3090에 최적화된 설정 생성 (하위 호환성)"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3090', 'kobart-summarization-large')
    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_rtx3090_baseline_config():
    """RTX 3090용 Baseline 설정 생성 (baseline.ipynb 기반)"""
    from .rtx3090_baseline_config import create_rtx3090_baseline_config as _create_baseline
    return _create_baseline()


def create_rtx3090_baseline_kfold_config(n_splits=5, ensemble_method='voting'):
    """RTX 3090용 Baseline + K-Fold 설정"""
    from .rtx3090_baseline_config import create_rtx3090_baseline_kfold_config as _create_kfold
    return _create_kfold(n_splits, ensemble_method)


def create_rtx3090_baseline_fast_config():
    """RTX 3090용 빠른 Baseline 설정 (실험용)"""
    from .rtx3090_baseline_config import create_rtx3090_baseline_fast_config as _create_fast
    return _create_fast()


def create_optimized_config_for_rtx4090():
    """RTX 4090에 최적화된 설정 생성 (하위 호환성)"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx4090', 'kobart-summarization-large')
    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_auto_optimized_config():
    """GPU를 자동 감지하여 최적화된 설정 생성"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config()
    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_custom_config(hardware: str, model: str, data_path: Optional[str] = None, **kwargs):
    """커스텀 설정 생성"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        hardware, model, data_path, **kwargs)
    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_lora_config(hardware: str, model: str, lora_type: str = 'auto', data_path: Optional[str] = None, **kwargs):
    """LoRA 설정으로 커스텀 설정 생성"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        hardware, model, data_path, **kwargs)

    # LoRA 설정 오버라이드
    if lora_type != 'auto':
        lora_preset = config_manager.presets.LORA_PRESETS.get(lora_type)
        if lora_preset:
            config['lora'] = {
                "enabled": lora_preset.enabled,
                "use_qlora": lora_preset.use_qlora,
                "r": lora_preset.r,
                "alpha": lora_preset.alpha,
                "dropout": lora_preset.dropout,
                "target_modules": lora_preset.target_modules or ["q_proj", "v_proj"],
                "bias": lora_preset.bias,
                "task_type": lora_preset.task_type,
            }
            config['general']['lora_preset'] = lora_type

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_qlora_config(hardware: str, model: str = 'bart-large', data_path: Optional[str] = None, **kwargs):
    """QLoRA 설정으로 큰 모델 학습 설정 생성"""
    return create_lora_config(hardware, model, 'qlora_standard', data_path, **kwargs)


def create_fast_config_for_rtx3060():
    """RTX 3060용 초고속 학습 설정 생성"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # 초고속 학습 설정
    config['training'].update({
        'num_train_epochs': 2,  # 매우 적은 에포크
        'learning_rate': 5e-5,  # 높은 학습률로 빠른 수렴
        'per_device_train_batch_size': 24,  # 최대 배치 크기
        'per_device_eval_batch_size': 24,
        'gradient_accumulation_steps': 1,  # accumulation 제거
        'eval_steps': 1000,  # 평가 빈도 대폭 감소
        'save_steps': 1000,
        'gradient_checkpointing': False,  # 속도 우선
        'dataloader_num_workers': 6,  # 최대 워커
        'fp16': True,  # Mixed precision 강제 활성화
    })

    # 짧은 시퀀스로 속도 향상
    config['tokenizer'].update({
        'encoder_max_len': 128,  # 매우 짧은 입력
        'decoder_max_len': 32,   # 매우 짧은 출력
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_high_quality_config_for_rtx3060():
    """RTX 3060용 고품질 Final Score 최적화 설정"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # 고품질 학습을 위한 설정
    config['training'].update({
        # 학습 파라미터 최적화
        'num_train_epochs': 8,  # 충분한 학습
        'learning_rate': 1.5e-5,  # 매우 낮은 학습률로 정밀 학습
        'per_device_train_batch_size': 8,  # 작은 배치로 안정성
        'per_device_eval_batch_size': 12,
        'gradient_accumulation_steps': 6,  # 효과적 배치 = 8*6=48

        # 스케줄링 최적화
        'warmup_ratio': 0.2,  # 긴 워밍업
        'lr_scheduler_type': 'polynomial',  # 부드러운 감소
        'polynomial_decay_power': 2.0,

        # 정규화 강화
        'weight_decay': 0.05,  # 강한 정규화
        'max_grad_norm': 0.3,  # 강한 그래디언트 클리핑
        'label_smoothing_factor': 0.1,  # 라벨 스무딩

        # 평가 및 저장 최적화
        'eval_steps': 200,  # 자주 평가
        'save_steps': 200,
        'early_stopping_patience': 8,  # 매우 긴 patience
        'early_stopping_threshold': 0.001,

        # 생성 품질 최대화
        'generation_num_beams': 8,  # 최대 beam
        'generation_length_penalty': 1.2,
        'generation_no_repeat_ngram_size': 4,
        'generation_do_sample': False,  # 결정적 생성

        # 메모리 최적화
        'gradient_checkpointing': True,
        'dataloader_pin_memory': True,
        'dataloader_num_workers': 2,  # 안정성 우선

        # 기타 최적화
        'fp16': True,
        'fp16_opt_level': 'O1',  # 안정적인 mixed precision
        'dataloader_drop_last': False,
        'ignore_data_skip': True,
    })

    # 토크나이저 고품질 설정
    config['tokenizer'].update({
        'encoder_max_len': 512,  # 긴 컨텍스트
        'decoder_max_len': 128,  # 긴 출력
        'padding': 'max_length',
        'truncation': True,
        'return_attention_mask': True,
        'return_token_type_ids': False,
    })

    # 추론 고품질 설정
    config['inference'].update({
        'num_beams': 8,
        'length_penalty': 1.2,
        'repetition_penalty': 1.02,
        'no_repeat_ngram_size': 4,
        'early_stopping': True,
        'do_sample': False,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0,
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


def create_balanced_config_for_rtx3060():
    """RTX 3060용 속도와 품질의 균형 잡힌 설정"""
    config_manager = ConfigManager()
    config = config_manager.create_optimized_config(
        'rtx3060', 'kobart-base-v2')

    # 균형 잡힌 설정
    config['training'].update({
        'num_train_epochs': 6,
        'learning_rate': 2.5e-5,
        'per_device_train_batch_size': 10,
        'per_device_eval_batch_size': 14,
        'gradient_accumulation_steps': 4,  # 효과적 배치 = 40

        'warmup_ratio': 0.12,
        'weight_decay': 0.03,
        'max_grad_norm': 0.4,

        'eval_steps': 250,
        'save_steps': 250,
        'early_stopping_patience': 6,

        'generation_num_beams': 6,
        'generation_length_penalty': 1.15,
        'generation_no_repeat_ngram_size': 3,

        'gradient_checkpointing': True,
        'dataloader_num_workers': 3,
    })

    config['tokenizer'].update({
        'encoder_max_len': 384,
        'decoder_max_len': 96,
    })

    config['inference'].update({
        'num_beams': 6,
        'length_penalty': 1.15,
        'repetition_penalty': 1.03,
        'no_repeat_ngram_size': 3,
    })

    config_manager.config = config
    config_manager.save_config(config)
    return config


if __name__ == "__main__":
    print("🧪 새로운 ConfigManager 테스트")
    print("=" * 60)

    config_manager = ConfigManager()

    # 1. 자동 최적화 설정
    print("\n1. 🚀 자동 하드웨어 감지 및 최적화:")
    auto_config = create_auto_optimized_config()

    # 2. 커스텀 설정 테스트
    print("\n2. 🎯 커스텀 설정 테스트:")

    # 다양한 조합 테스트
    test_combinations = [
        ('rtx3060', 'kobart-base-v2'),
        ('rtx3090', 'kobart-summarization-large'),
        ('rtx4090', 'bart-large'),
        ('cpu', 'kobart-base-v2'),
    ]

    for hardware, model in test_combinations:
        print(f"\n   {hardware.upper()} + {model}:")
        try:
            config = create_custom_config(hardware, model)
            training_config = config['training']
            tokenizer_config = config['tokenizer']

            print(
                f"     배치 크기: {training_config['per_device_train_batch_size']}")
            print(f"     학습률: {training_config['learning_rate']}")
            print(
                f"     시퀀스 길이: {tokenizer_config['encoder_max_len']}/{tokenizer_config['decoder_max_len']}")
            print(f"     에포크: {training_config['num_train_epochs']}")
            print(f"     FP16: {training_config['fp16']}")
            print(
                f"     Gradient Checkpointing: {training_config['gradient_checkpointing']}")
        except Exception as e:
            print(f"     ❌ 오류: {e}")

    # 3. 하드웨어 사양 정보
    print("\n3. 📊 지원되는 하드웨어 사양:")
    for key, spec in ConfigPresets.HARDWARE_SPECS.items():
        print(f"   {key}: {spec.name} ({spec.vram_gb}GB VRAM)")

    # 4. 모델 사양 정보
    print("\n4. 🤖 지원되는 모델 사양:")
    for key, spec in ConfigPresets.MODEL_SPECS.items():
        print(
            f"   {key}: {spec.name} ({spec.parameters}M params, {spec.recommended_vram}GB 권장)")

    print("\n✅ 새로운 ConfigManager 테스트 완료!")
    print("\n💡 사용 예시:")
    print("   # 자동 최적화")
    print("   config = create_auto_optimized_config()")
    print("   # 커스텀 설정")
    print("   config = create_custom_config('rtx3090', 'kobart-summarization-large')")
    print("   # 특정 GPU 설정")
    print("   config = create_optimized_config_for_rtx4090()")


def create_kfold_config_for_rtx3060(n_splits=5, ensemble_method='voting'):
    """RTX 3060용 K-Fold 교차 검증 설정 생성"""
    config_manager = ConfigManager()

    # 기본 고품질 설정을 베이스로 사용
    config = create_high_quality_config_for_rtx3060()

    # K-Fold 설정 활성화 및 조정
    config['kfold'] = {
        'enabled': True,
        'n_splits': n_splits,
        'stratified': True,  # 텍스트 길이 기반 계층화
        'random_state': 42,
        'ensemble_method': ensemble_method,
        'save_individual_models': True,
        'use_ensemble_inference': True
    }

    # K-Fold에 최적화된 학습 설정 조정
    config['training'].update({
        'num_train_epochs': 8,  # fold별로 조금 줄임
        'early_stopping_patience': 2,  # 더 빠른 조기 종료
        'save_total_limit': 2,  # 저장 공간 절약
        'evaluation_strategy': 'steps',
        'eval_steps': 100,
        'save_steps': 100,
        'logging_steps': 50,
        'metric_for_best_model': 'eval_final_score',
        'load_best_model_at_end': True,
        'greater_is_better': True
    })

    # 메모리 최적화 (여러 모델 학습을 위해)
    config['training'].update({
        'per_device_train_batch_size': 8,  # 배치 크기 줄임
        'per_device_eval_batch_size': 8,
        'gradient_accumulation_steps': 4,  # 그래디언트 누적으로 보상
        'dataloader_num_workers': 2,
        'dataloader_pin_memory': False
    })

    # LoRA 설정 최적화
    if 'lora' in config:
        config['lora'].update({
            'r': 16,  # 적절한 rank
            'alpha': 32,
            'dropout': 0.1,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2']
        })

    print("🔄 RTX 3060용 K-Fold 교차 검증 설정이 생성되었습니다.")
    print(f"   - Fold 수: {n_splits}")
    print(f"   - 앙상블 방법: {ensemble_method}")
    print(f"   - 계층화: 활성화")
    print(f"   - LoRA: 활성화")

    return config


def create_kfold_config_for_high_performance(n_splits=10, ensemble_method='weighted'):
    """고성능 GPU용 K-Fold 교차 검증 설정 생성"""
    config_manager = ConfigManager()

    # 고성능 설정을 베이스로 사용
    config = config_manager.create_optimized_config(
        hardware_key='rtx4090',
        model_key='bart-large'
    )

    # K-Fold 설정
    config['kfold'] = {
        'enabled': True,
        'n_splits': n_splits,
        'stratified': True,
        'random_state': 42,
        'ensemble_method': ensemble_method,
        'save_individual_models': True,
        'use_ensemble_inference': True
    }

    # 고성능 학습 설정
    config['training'].update({
        'num_train_epochs': 12,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'gradient_accumulation_steps': 2,
        'learning_rate': 5e-6,  # 더 낮은 학습률로 안정성 확보
        'warmup_ratio': 0.05,
        'early_stopping_patience': 3,
        'evaluation_strategy': 'steps',
        'eval_steps': 200,
        'save_steps': 200,
        'logging_steps': 100
    })

    print("🚀 고성능 GPU용 K-Fold 교차 검증 설정이 생성되었습니다.")
    print(f"   - Fold 수: {n_splits}")
    print(f"   - 앙상블 방법: {ensemble_method}")
    print(f"   - Full Fine-tuning 사용")

    return config


def create_fast_kfold_config_for_rtx3060(n_splits=3, ensemble_method='best'):
    """RTX 3060용 빠른 K-Fold 설정 생성 (실험용)"""
    config_manager = ConfigManager()

    # 빠른 설정을 베이스로 사용
    config = create_balanced_config_for_rtx3060()

    # 빠른 K-Fold 설정
    config['kfold'] = {
        'enabled': True,
        'n_splits': n_splits,  # 적은 fold 수
        'stratified': False,  # 계층화 비활성화로 속도 향상
        'random_state': 42,
        'ensemble_method': ensemble_method,  # 최고 모델만 사용
        'save_individual_models': False,  # 저장 공간 절약
        'use_ensemble_inference': False  # 단일 모델 추론
    }

    # 메모리 안전 학습 설정 (RTX 3060 노트북 최적화)
    config['training'].update({
        'num_train_epochs': 4,  # 더 적은 에포크
        'per_device_train_batch_size': 6,  # 배치 크기 대폭 감소
        'per_device_eval_batch_size': 6,
        'gradient_accumulation_steps': 4,  # 그래디언트 누적으로 보상
        'early_stopping_patience': 1,  # 매우 빠른 조기 종료
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'logging_steps': 20,
        'dataloader_num_workers': 0,  # 멀티프로세싱 비활성화
        'dataloader_pin_memory': False,  # 메모리 핀 비활성화
        'remove_unused_columns': True,  # 불필요한 컬럼 제거
        'fp16': True,  # FP16 사용으로 메모리 절약
        'gradient_checkpointing': True,  # 그래디언트 체크포인팅으로 메모리 절약
        'max_grad_norm': 1.0,  # 그래디언트 클리핑
    })

    # 토크나이저 설정도 메모리 절약
    config['tokenizer'].update({
        'encoder_max_len': 256,  # 시퀀스 길이 단축
        'decoder_max_len': 64,   # 디코더 길이 단축
    })

    print("⚡ RTX 3060용 빠른 K-Fold 설정이 생성되었습니다.")
    print(f"   - Fold 수: {n_splits} (빠른 실험용)")
    print(f"   - 앙상블 방법: {ensemble_method}")
    print(f"   - 최적화: 속도 우선")

    return config
