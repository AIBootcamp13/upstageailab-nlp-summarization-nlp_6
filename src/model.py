import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def load_model_and_tokenizer(config: dict):
    """
    모델을 로드하기 전에 config를 업데이트하고 ignore_mismatched_sizes=True를 사용하여
    size mismatch 및 meta tensor 오류를 원천적으로 방지합니다.
    """
    model_type = config['model']['type']
    model_name = config['model']['architectures'][model_type]['name']
    use_lora = config.get('lora', {}).get('use_lora', False)

    is_causal_lm = 'koalpaca' in model_type
    model_class = AutoModelForCausalLM if is_causal_lm else AutoModelForSeq2SeqLM
    
    print(f"Model type detected: {'CausalLM' if is_causal_lm else 'Seq2Seq'}. Using class: {model_class.__name__}")

    # 1. 토크나이저를 먼저 로드하고 스페셜 토큰을 추가하여 최종 어휘 크기를 확정합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if is_causal_lm and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer's pad_token is set to eos_token.")
    if config['model']['special_tokens']:
        tokenizer.add_special_tokens({'additional_special_tokens': config['model']['special_tokens']})
        print(f"Special tokens added. New tokenizer size: {len(tokenizer)}")

    # 2. 모델 설정을 불러와서 토크나이저의 최종 크기로 업데이트합니다.
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.vocab_size = len(tokenizer)

    # 3. 모델 로딩 인자를 준비합니다.
    model_args = {
        "config": model_config,
        "device_map": "auto",
        "ignore_mismatched_sizes": True  # ✨ [핵심 수정] 크기가 다른 임베딩 레이어는 무시하고 로드하도록 설정
    }
    if use_lora:
        print("QLoRA is enabled. Loading model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_args["quantization_config"] = quantization_config
    else:
        print("Loading model for full fine-tuning...")

    # 4. 업데이트된 설정을 사용하여 모델을 로드합니다.
    # 이제 모델이 처음부터 올바른 크기로 생성됩니다.
    model = model_class.from_pretrained(model_name, **model_args)
    print(f"Model loaded with final vocab size: {model.config.vocab_size}")

    # 이제 더 이상 resize나 tie_weights를 호출할 필요가 없습니다.

    if use_lora:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training.")
    
    return model, tokenizer
