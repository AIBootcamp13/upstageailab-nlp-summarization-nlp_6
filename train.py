import os
from numpy import VisibleDeprecationWarning
import torch
import wandb
from datetime import datetime
from src.utils import load_config
from src.data_loader import DataPreprocessor
from src.model import load_model_and_tokenizer
from src.trainer import get_trainer
from create_dummy_data import create_data_file
import warnings

def main():
    # 1. 설정 파일 로드
    config_path = "config/config.yaml"
    config = load_config(config_path)
    
    # 선택된 모델 타입 가져오기
    model_type = config['model']['type']
    
    # 해당 모델 타입에 맞는 상세 설정 가져오기
    model_specific_config = config['model']['architectures'].get(model_type)
    
    if not model_specific_config:
        raise ValueError(f"Configuration for model type '{model_type}' not found in config.yaml under 'architectures'.")

    # 모델별 기본 정보 및 LoRA 설정 병합
    config['model'].update({k: v for k, v in model_specific_config.items() if k != 'lora'})
    if 'lora' in model_specific_config:
        config['lora'].update(model_specific_config['lora'])

    # use_lora 플래그에 따라 옵티마이저 자동 변경
    use_lora = config.get('lora', {}).get('use_lora', False)
    # if use_lora:
    #     print("LoRA training is enabled. Optimizer will be set to 'paged_adamw_8bit'.")
    #     config['training']['optim'] = 'paged_adamw_8bit'
    # else:
    #     print("Full parameter fine-tuning is enabled. Optimizer is set to 'adamw_torch'.")
    #     config['training']['optim'] = 'adamw_torch'

    # WandB 이름 자동 생성
    if not config['wandb']['name']:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        lora_tag = "LoRA" if use_lora else "Full"
        config['wandb']['name'] = f"{model_type}-{lora_tag}-{timestamp}"
        
    print("--- Dynamically Merged Configuration ---")
    print(f"Model Name: {config['model']['name']}")
    print(f"LoRA Settings: {config['lora']}")
    print("-" * 40)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 테스트 모드일 경우 더미 데이터 생성
    if config.get('test_mode', False):
        print("Test mode enabled. Generating dummy data...")
        create_data_file('train_test.csv', 100)
        create_data_file('val_test.csv', 20)
        print("Dummy data generation complete.")

    model, tokenizer = load_model_and_tokenizer(config)
    # QLoRA 사용 시 model.to(device)는 device_map="auto"로 대체되므로 주석 처리
    # model.to(device) 

    preprocessor = DataPreprocessor(config, tokenizer)
    train_dataset, val_dataset = preprocessor.setup_datasets()

    # 데이터셋 크기 축소 (테스트용)
    if config.get('test_mode', False):
        train_dataset.data = {k: v[:min(len(train_dataset), 50)] for k, v in train_dataset.data.items()}
        train_dataset.data_len = len(train_dataset.data['input_ids'])
        val_dataset.data = {k: v[:min(len(val_dataset), 10)] for k, v in val_dataset.data.items()}
        val_dataset.data_len = len(val_dataset.data['input_ids'])
    
    print(f"Train dataset size (sampled): {len(train_dataset)}")
    print(f"Validation dataset size (sampled): {len(val_dataset)}")

    trainer = get_trainer(config, model, tokenizer, train_dataset, val_dataset)

    print("Training started...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=VisibleDeprecationWarning)
        trainer.train()
    print("Training finished.")

    best_model_path = os.path.join(config['output_dir'], "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model and tokenizer saved to {best_model_path}")
    
    if config['training']['report_to'] == 'wandb':
        wandb.finish()

if __name__ == "__main__":
    main()
