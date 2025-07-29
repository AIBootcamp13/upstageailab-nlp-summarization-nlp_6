# create_dummy_data.py

import os
import pandas as pd
import random
from faker import Faker

# 한국어 가짜 데이터 생성을 위한 Faker 객체 초기화
fake = Faker('ko_KR')

def generate_dialogue(num_turns: int = 6) -> str:
    """
    #Person1#과 #Person2#가 대화하는 형식의 가짜 대화문을 생성합니다.
    """
    dialogue_parts = []
    persons = ["#Person1#", "#Person2#"]
    for _ in range(num_turns):
        speaker = random.choice(persons)
        sentence = fake.sentence(nb_words=random.randint(8, 15))
        dialogue_parts.append(f"{speaker}: {sentence}")
    
    return "\n".join(dialogue_parts)

def create_data_file(filename: str, num_samples: int, is_test: bool = False):
    """
    지정된 수의 샘플을 가진 더미 데이터 파일을 생성합니다.
    """
    print(f"Generating {filename} with {num_samples} samples...")
    
    data = []
    file_prefix = filename.split('.')[0] # 'train', 'dev', 'test'
    
    for i in range(num_samples):
        fname = f"{file_prefix}_{i}"
        dialogue = generate_dialogue(random.randint(4, 10))
        
        if is_test:
            data.append({"fname": fname, "dialogue": dialogue})
        else:
            summary = fake.sentence(nb_words=random.randint(10, 20))
            data.append({"fname": fname, "dialogue": dialogue, "summary": summary})
            
    df = pd.DataFrame(data)
    
    # data 폴더 생성
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # CSV 파일로 저장
    df.to_csv(os.path.join(data_dir, filename), index=False, encoding='utf-8')
    print(f"Successfully created {os.path.join(data_dir, filename)}")

if __name__ == "__main__":
    # 테스트에 필요한 만큼의 작은 데이터셋 생성
    num_train_samples = 100
    num_dev_samples = 20
    num_test_samples = 20
    
    create_data_file('train.csv', num_train_samples)
    create_data_file('dev.csv', num_dev_samples)
    create_data_file('test.csv', num_test_samples, is_test=True)
    
    print("\nDummy data generation complete!")