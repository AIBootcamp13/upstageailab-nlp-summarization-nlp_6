import pandas as pd
from transformers import AutoTokenizer

data_path = './data'
# 데이터와 토크나이저 불러오기
df = pd.read_csv(f"{data_path}/train.csv") # 실제 파일 경로로 수정
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-base-ko") # 사용하는 모델 이름으로 수정

# 각 대화의 토큰 길이 계산
token_lengths = [len(tokenizer.encode(text)) for text in df['dialogue']]
df['token_length'] = token_lengths

# 길이 통계 확인 (90%, 95% 지점을 보는 것이 중요)
print(df['token_length'].describe(percentiles=[.75, .90, .95]))

# 요약문('summary' 컬럼)의 토큰 길이 계산
summary_lengths = [len(tokenizer.encode(text)) for text in df['summary']]
df['summary_length'] = summary_lengths

# 길이 통계 확인 (95%, 99% 지점을 보는 것이 중요)
print(df['summary_length'].describe(percentiles=[.95, .99]))