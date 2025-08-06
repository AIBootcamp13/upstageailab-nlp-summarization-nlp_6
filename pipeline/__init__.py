"""
대화 요약 파이프라인 패키지
"""

from .config_manager import (
    ConfigManager,
    create_optimized_config_for_rtx3060,
    create_optimized_config_for_rtx3090,
    create_optimized_config_for_rtx4090,
    create_auto_optimized_config,
    create_custom_config,
    create_lora_config,
    create_qlora_config,
    create_kfold_config_for_rtx3060,
    create_kfold_config_for_high_performance,
    create_fast_kfold_config_for_rtx3060,
    create_rtx3090_baseline_config,
    create_rtx3090_baseline_kfold_config,
    create_rtx3090_baseline_fast_config
)
from .data_processor import DataProcessor, Preprocess
from .model_manager import ModelManager, TrainingManager
from .inference_manager import InferenceManager, InteractiveInference
from .kfold_manager import KFoldManager

__all__ = [
    'ConfigManager',
    'create_optimized_config_for_rtx3060',
    'create_optimized_config_for_rtx3090',
    'create_optimized_config_for_rtx4090',
    'create_auto_optimized_config',
    'create_custom_config',
    'create_lora_config',
    'create_qlora_config',
    'create_kfold_config_for_rtx3060',
    'create_kfold_config_for_high_performance',
    'create_fast_kfold_config_for_rtx3060',
    'create_rtx3090_baseline_config',
    'create_rtx3090_baseline_kfold_config',
    'create_rtx3090_baseline_fast_config',
    'DataProcessor',
    'Preprocess',
    'ModelManager',
    'TrainingManager',
    'InferenceManager',
    'InteractiveInference',
    'KFoldManager'
]

__version__ = "1.0.0"
