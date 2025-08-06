import pandas as pd

# 결과 파일 확인
df = pd.read_csv('prediction/output.csv')
print(f'총 샘플 수: {len(df)}')

# 누락된 샘플 확인
expected = set(f"test_{i}" for i in range(499))
actual = set(df["fname"])
missing = expected - actual

print(f'누락된 샘플 수: {len(missing)}')
if missing:
    print(f'누락된 샘플: {sorted(list(missing))[:10]}...')  # 처음 10개만 표시
else:
    print('✅ 모든 샘플이 생성되었습니다!')

# 샘플 확인
print(f'\n=== 결과 샘플 ===')
for i in range(min(5, len(df))):
    print(f'{df.iloc[i]["fname"]}: {df.iloc[i]["summary"][:100]}...')