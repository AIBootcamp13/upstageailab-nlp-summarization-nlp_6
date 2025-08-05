# inference.py

import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from peft import PeftModel

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 1. 모델 및 토크나이저 로딩
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/qwen3-4b-base-unsloth-bnb-4bit", # Base model
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add special tokens (same as in finetune_summarizer_3090.py)
special_tokens = ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#DateOfBirth#', '#PassportNumber#', '#SSN#', '#CardNumber#', '#CarNumber#', '#Email#']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))

# Load the LoRA adapter
model = PeftModel.from_pretrained(
    model, 
    "Qwen3-4B-Base_summarizer_lora",
    is_trainable=False,       # inference 전용
)


# 2. 제출용 CSV 파일 생성
print("\n\n===== Generating submission.csv =====")

# 테스트 데이터 로드
test_dataset = load_dataset('csv', data_files='/data/ephemeral/home/dev/upstageailab-nlp-summarization-nlp_6/data/test.csv', split='train')

# 요약 생성
fnames = []
summaries = []

for item in tqdm(test_dataset):
    fname = item['fname']
    dialogue = item['dialogue']
    
    # 프롬프트 생성
    prompt = alpaca_prompt.format(
        "Summarize the following dialogue.", # instruction
        dialogue, # input
        "", # output - 비워둠
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # 요약 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=False, # beam_search와 호환성 문제로 False로 설정
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_beams=4,
    )
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # 응답 부분만 추출
    try:
        response_part = decoded_output.split("### Response:")[1].strip()
    except IndexError:
        response_part = decoded_output # 응답 형식이 다를 경우 전체 출력

    fnames.append(fname)
    summaries.append(response_part)

# DataFrame 생성
submission_df = pd.DataFrame({
    'fname': fnames,
    'summary': summaries
})

# CSV 파일로 저장
submission_df.to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' created successfully.")
