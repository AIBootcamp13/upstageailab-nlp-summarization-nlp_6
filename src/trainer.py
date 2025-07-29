import os
import wandb
import inspect
from rouge import Rouge
from transformers import (
    TrainingArguments, 
    Trainer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

def compute_metrics(eval_pred, tokenizer, config):
    """
    Seq2Seq 및 Causal LM 모델 모두에 대해 ROUGE 점수를 계산하는 함수.
    """
    predictions, labels = eval_pred
    
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
    
    tokens_to_remove = ['<s>', '</s>', '<pad>', '<usr>'] 
    
    def clean_text(text_list):
        cleaned_list = []
        for text in text_list:
            for token in tokens_to_remove:
                text = text.replace(token, "")
            cleaned_list.append(text.strip())
        return cleaned_list

    cleaned_preds = clean_text(decoded_preds)
    cleaned_labels = clean_text(decoded_labels)

    model_type = config['model']['type']
    is_causal_lm = 'koalpaca' in model_type
    
    if is_causal_lm:
        cleaned_labels = [label.split("### 요약:")[1].strip() if "### 요약:" in label else label for label in cleaned_labels]
        cleaned_preds = [pred.split("### 요약:")[1].strip() if "### 요약:" in pred else pred for pred in cleaned_preds]

    final_preds = []
    final_labels = []
    for pred, label in zip(cleaned_preds, cleaned_labels):
        if label:
            final_preds.append(pred if pred else " ")
            final_labels.append(label)

    if not final_labels:
        return {}
        
    rouge = Rouge()
    scores = rouge.get_scores(final_preds, final_labels, avg=True)
    result = {key: value['f'] for key, value in scores.items()}

    print("\n--- Sample Predictions (Final) ---")
    for i in range(min(3, len(final_preds))):
        print(f"Pred: {final_preds[i]}")
        print(f"Gold: {final_labels[i]}")

    return result

def get_trainer(config: dict, model, tokenizer, train_dataset, val_dataset):
    """
    모델 아키텍처(Seq2Seq/CausalLM) 및 LoRA 적용 여부에 따라
    적절한 Trainer를 설정하고 반환합니다.
    """
    model_type = config['model']['type']
    is_causal_lm = 'koalpaca' in model_type
    
    use_lora = config.get('lora', {}).get('use_lora', False)
    if use_lora:
        print("Applying LoRA to the model...")
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['model']['architectures'][model_type]['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM if is_causal_lm else TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, lora_config)
        
        if config['model']['special_tokens']:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            print(f"Resized token embeddings to: {len(tokenizer)}")

        print("\n--- LoRA Applied Model ---")
        model.print_trainable_parameters()
        print("-" * 26 + "\n")
    else:
        if config['model']['special_tokens']:
             model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        print("Skipping LoRA application. Proceeding with full fine-tuning.")

    if config['training']['report_to'] == 'wandb':
        from dotenv import load_dotenv
        load_dotenv()
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            config=config
        )
        os.environ["WANDB_LOG_MODEL"] = "true"
        os.environ["WANDB_WATCH"] = "false"

    if is_causal_lm:
        print("Using standard Trainer for Causal LM.")
        training_args_class = TrainingArguments
        trainer_class = Trainer
    else:
        print("Using Seq2SeqTrainer for Seq2Seq model.")
        training_args_class = Seq2SeqTrainingArguments
        trainer_class = Seq2SeqTrainer

    training_params = config['training'].copy()
    
    early_stopping_patience = training_params.pop("early_stopping_patience", 3)
    early_stopping_threshold = training_params.pop("early_stopping_threshold", 0.0)

    args_dict = {
        "output_dir": config['output_dir'],
        "logging_dir": config['logging_dir'],
        "overwrite_output_dir": True,
        **training_params
    }

    if is_causal_lm:
        args_dict.pop("predict_with_generate", None)
        args_dict.pop("generation_max_length", None)
        args_dict.pop("metric_for_best_model", None)
    else:
        args_dict["predict_with_generate"] = True
        args_dict["generation_max_length"] = training_params.get('generation_max_length', 128)
        args_dict["metric_for_best_model"] = "rouge-l"

    training_args = training_args_class(**args_dict)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )
    
    # ✨ [핵심 수정] device_map="auto" 사용 시 Trainer가 모델을 이동시키려는 동작을 원천적으로 차단합니다.
    if getattr(model, "hf_device_map", None) is not None:
        print("Model is already on multiple devices. Disabling Trainer's model movement.")
        def _move_model_to_device_noop(self, model, device):
            return model
        trainer_class._move_model_to_device = _move_model_to_device_noop
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, config),
        callbacks=[early_stopping]
    )
    
    return trainer
