import os
from numpy import VisibleDeprecationWarning
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader
from src.utils import load_config
from src.data_loader import DataPreprocessor, SummarizationDataset
import warnings

def inference():
    """
    학습 방식(QLoRA 또는 풀 파인튜닝)에 따라 모델을 로드하여 테스트 데이터의 요약문을 생성하고,
    결과를 submission.csv 파일로 저장합니다.
    """
    # 1. 설정 및 장치 준비
    config_path = "config/config.yaml"
    config = load_config(config_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1.5. LoRA 사용 여부 확인
    use_lora = config.get('lora', {}).get('use_lora', False)
    checkpoint_path = config['inference']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train a model first.")

    # 2. 모델 및 토크나이저 로드 (조건부)
    if use_lora:
        # --- QLoRA로 학습된 모델을 로드하는 경우 ---
        print(f"LoRA is enabled. Loading base model and adapter from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model_name = config['model']['architectures'][config['model']['type']]['name']
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
        print("LoRA model loaded and merged successfully.")
        
    else:
        # --- 풀 파인튜닝된 모델을 로드하는 경우 ---
        print(f"LoRA is disabled. Loading full fine-tuned model from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            local_files_only=True,
            device_map="auto"
        )
        model.resize_token_embeddings(len(tokenizer))
        print("Full fine-tuned model loaded successfully.")

    # 3. 데이터 준비
    preprocessor = DataPreprocessor(config, tokenizer)
    test_path = os.path.join(config['data_dir'], 'test.csv')
    tokenized_test, fnames = preprocessor.prepare_for_inference(test_path)
    test_dataset = SummarizationDataset(tokenized_test)
    test_dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'])

    # 4. 요약문 생성
    summaries = []
    model.eval()
    with torch.no_grad():
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=VisibleDeprecationWarning)
        
        for batch in tqdm(test_dataloader, desc="Generating summaries"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['model']['decoder_max_len'],
                num_beams=config['inference']['num_beams'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping']
            )
            
            # ✨✨✨ [수정 1] skip_special_tokens=False로 변경하여 모든 토큰을 텍스트로 변환합니다.
            decoded_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            summaries.extend(decoded_summaries)

    # 5. 후처리 및 제출 파일 생성
    if len(fnames) != len(summaries):
        raise ValueError(f"Length mismatch! Got {len(fnames)} filenames and {len(summaries)} summaries.")

    # ✨✨✨ [수정 2] 수동으로 불필요한 기술적 토큰만 제거합니다.
    # config 파일에 정의된 remove_tokens ('<s>', '</s>' 등)만 사용합니다.
    tokens_to_remove = config['inference'].get('remove_tokens', [])
    cleaned_summaries = []
    for s in summaries:
        for token in tokens_to_remove:
            s = s.replace(token, "")
        cleaned_summaries.append(s.strip())
        
    submission = pd.DataFrame({'fname': fnames, 'summary': cleaned_summaries})
    
    result_dir = config['inference']['result_path']
    os.makedirs(result_dir, exist_ok=True)
    submission_path = os.path.join(result_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False, encoding='utf-8')
    
    print(f"\nSubmission file saved to {submission_path}")
    print("\n--- Sample Submission ---")
    print(submission.head())

if __name__ == "__main__":
    try:
        inference()
    except Exception as e:
        print(f"An error occurred during inference: {e}")
    finally:
        # GPU 캐시 정리 (선택 사항)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nGPU cache cleared.")
