import json
import os
from transformers import AutoTokenizer

def debug_checkpoint(path: str):
    """
    저장된 모델 체크포인트의 설정과 토크나이저의 상태를 비교하여
    'size mismatch' 오류의 원인을 진단합니다.
    """
    print(f"--- 🕵️‍♂️ Debugging Checkpoint at: {path} ---")

    # 1. 모델 설정 파일(config.json) 확인
    config_file_path = os.path.join(path, "config.json")
    vocab_size_in_config = "N/A"
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        vocab_size_in_config = model_config.get("vocab_size")
        print(f"\n[1] Found 'config.json'.")
        print(f"    - 'vocab_size' in config.json: {vocab_size_in_config}")
    else:
        print("\n[1] 'config.json' not found in the checkpoint directory.")

    # 2. 토크나이저(tokenizer) 확인
    vocab_size_in_tokenizer = "N/A"
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        vocab_size_in_tokenizer = len(tokenizer)
        print(f"\n[2] Loaded tokenizer from checkpoint.")
        print(f"    - vocab_size from len(tokenizer): {vocab_size_in_tokenizer}")
    except Exception as e:
        print(f"\n[2] Failed to load tokenizer from checkpoint. Error: {e}")
        
    # 3. 최종 진단
    print("\n--- 🩺 Final Diagnosis ---")
    if vocab_size_in_config == vocab_size_in_tokenizer:
        print("✅ The vocab sizes seem to match. The problem might be more subtle.")
    else:
        print(f"❌ MISMATCH FOUND! The model config says {vocab_size_in_config}, but the tokenizer has {vocab_size_in_tokenizer} tokens.")
        print("This confirms the root cause of the 'size mismatch' error.")
    print("-" * 40)


if __name__ == "__main__":
    # ❗️이 경로를 실제 best_model이 저장된 경로로 수정해주세요.
    checkpoint_path = "results/best_model" 
    
    if not os.path.isdir(checkpoint_path):
        print(f"Error: Directory not found at '{checkpoint_path}'. Please make sure the path is correct.")
    else:
        debug_checkpoint(checkpoint_path)