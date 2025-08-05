# finetune_summarizer_3090.py

import torch
import numpy as np
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from rouge import Rouge
from typing import Any, Dict, List, Optional, Tuple, Union

# 1. 모델 및 토크나이저 로딩
# RTX 3090은 bfloat16을 지원하므로 성능 향상을 기대할 수 있습니다.
max_seq_length = 2048  # 24GB VRAM으로 4096까지 시도해볼 수 있습니다.
eval_max_seq_length = 512  # 평가 시 생성될 최대 토큰 길이
dtype = None  # None으로 두면 unsloth가 자동 선택 (bfloat16 지원 시 우선 사용)
load_in_4bit = True  # 4bit 양자화로 메모리 사용량 대폭 감소

# unsloth가 제공하는 4bit 양자화된 Llama-3 모델 로딩
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 개인정보 관련 스페셜 토큰 추가
special_tokens = ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#DateOfBirth#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#', '#Email#']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))

# 2. LoRA 설정 및 모델 패치
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank. 추천값: 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 학습 가능한 파라미터 수 확인
model.print_trainable_parameters()

# 3. 데이터 준비 (Data Preprocessing)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = [f"Summarize the following dialogue about {topic}." for topic in examples["topic"]]
    inputs = examples["dialogue"]
    outputs = examples["summary"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }

# train.csv 파일을 로드합니다.
dataset = load_dataset('csv', data_files='/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/train.csv', split='train')
eval_dataset = load_dataset('csv', data_files='/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/dev.csv', split='train')

# 필요한 컬럼만 남기고, 컬럼명을 instruction, input, output으로 변경합니다.
dataset = dataset.map(
    lambda x: {
        "instruction": "Summarize the following dialogue.",
        "input": x["dialogue"],
        "output": x["summary"]
    },
)

dataset = dataset.map(formatting_prompts_func, batched=True,)
eval_dataset = eval_dataset.map(
    lambda x: {
        "instruction": f"Summarize the following dialogue about {x['topic']}.",
        "input": x["dialogue"],
        "output": x["summary"]
    },
)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True,)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predict_with_generate=True를 사용하면 predictions는 생성된 토큰 ID입니다.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge-L의 F1 점수를 계산합니다.
    rouge = Rouge()
    # 생성된 텍스트와 레이블에서 응답 부분만 추출하여 비교합니다.
    cleaned_preds = [pred.split("### Response:")[1].strip() if "### Response:" in pred else "" for pred in decoded_preds]
    cleaned_labels = [label.split("### Response:")[1].strip() if "### Response:" in label else "" for label in decoded_labels]

    # 비어있는 예측이나 레이블이 있을 경우 점수 계산에서 제외합니다.
    cleaned_preds = [pred for pred, label in zip(cleaned_preds, cleaned_labels) if pred and label]
    cleaned_labels = [label for pred, label in zip(cleaned_preds, cleaned_labels) if pred and label]

    if not cleaned_preds or not cleaned_labels:
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "final_score": 0}
        
    scores = rouge.get_scores(cleaned_preds, cleaned_labels, avg=True)
    
    result = {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }

    # final_score 계산
    result["final_score"] = result["rouge-1"] + result["rouge-2"] + result["rouge-l"]
    return result

# 커스텀 트레이너 정의: 평가 시 generate에 eval_max_seq_length를 max_new_tokens로 적용
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # predict_with_generate를 강제로 활성화 (TrainingArguments에 없으므로 내부적으로 설정)
        self.predict_with_generate = True

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_new_tokens": eval_max_seq_length,
            "use_cache": True,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            # 필요 시 다른 생성 파라미터 추가 (기본값으로 설정)
        }

        with torch.no_grad():
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **gen_kwargs
            )

        # 배치 크기 맞추기
        if generated_tokens.shape[0] != inputs["input_ids"].shape[0]:
            generated_tokens = generated_tokens[: inputs["input_ids"].shape[0]]

        labels = inputs.get("labels", None)
        return (None, generated_tokens, labels)

# 4. 모델 학습 (Training)
trainer = CustomSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  # eval을 위해 추가
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    args=TrainingArguments(
        # RTX 3090 (24GB)에 맞게 배치 사이즈 및 그래디언트 축적 단계 조정
        # VRAM이 충분하므로 배치 사이즈를 늘려 학습 속도를 높일 수 있습니다.
        per_device_train_batch_size=4,  # 2 -> 4 또는 8로 상향 조정
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,  # (4 * 4 = 16)의 유효 배치 사이즈. 2 또는 1로 줄여도 좋습니다.
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-4,
        # RTX 3090은 bf16을 지원하므로 fp16 대신 bf16을 사용합니다.
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="final_score",
        greater_is_better=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        max_grad_norm=1.0,
    ),
)

# 학습 시작
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
trainer_stats = trainer.train()

# 5. 추론 (Inference)
print("\n\n===== Inference Test =====")
instruction = "Summarize the following dialogue."
input_text = """
Emily: Hi, I'm thinking of visiting your city next month. Any recommendations for must-see places?
David: Absolutely! You should definitely check out the old town. It's full of historic buildings and charming cobblestone streets.
Emily: That sounds lovely! What about museums? I'm a big fan of art and history.
David: Then you can't miss the National Art Gallery. It has an amazing collection of both classic and contemporary art. And for history, the City History Museum is fantastic.
Emily: Great! I'll add them to my list. Any tips for local food?
David: You have to try the seafood paella at 'The Salty Anchor' by the harbor. It's the best in town!
Emily: Perfect! Thanks so much for the suggestions, David!
David: You're welcome! Have a wonderful trip.
"""

prompt = alpaca_prompt.format(instruction, input_text, "")

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# FastLanguageModel은 내부적으로 추론을 위한 최적화를 수행합니다.
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("=" * 50)
print("Model Inference Result:")
# 응답 부분만 깔끔하게 출력하기
response_part = decoded_output[0].split("### Response:")[1].strip()
print(response_part)
print("=" * 50)

# 6. 모델 저장
model.save_pretrained("Qwen3-4B-Base_summarizer_lora")
tokenizer.save_pretrained("Qwen3-4B-Base_summarizer_lora")

print("Fine-tuning complete and model saved to 'Qwen3-4B-Base_summarizer_lora'.")
