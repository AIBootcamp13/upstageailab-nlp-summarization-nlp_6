"""
데이터 처리 모듈
"""

import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Preprocess:
    """데이터 전처리 클래스"""
    def __init__(self, bos_token: str, eos_token: str):
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path, is_train=True):
        """CSV 파일을 데이터프레임으로 변환"""
        df = pd.read_csv(file_path)
        if is_train:
            return df[['fname', 'dialogue', 'summary']]
        else:
            return df[['fname', 'dialogue']]

    def make_input(self, dataset, is_test=False):
        """모델 입력 형태로 데이터 변환"""
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x: self.bos_token + str(x))
            decoder_output = dataset['summary'].apply(lambda x: str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()

class DatasetForTrain(Dataset):
    """학습용 데이터셋 클래스"""
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.length

class DatasetForVal(Dataset):
    """검증용 데이터셋 클래스"""
    def __init__(self, encoder_input, decoder_input, labels, length):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()}
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2)
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return self.length

class DatasetForInference(Dataset):
    """추론용 데이터셋 클래스"""
    def __init__(self, encoder_input, test_id, length):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.length = length

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.length

class DataProcessor:
    """데이터 처리 메인 클래스"""
    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocess(
            config['tokenizer']['bos_token'],
            config['tokenizer']['eos_token']
        )

    def prepare_train_dataset(self, tokenizer):
        """학습용 데이터셋 준비"""
        data_path = self.config['general']['data_path']

        print(f"📁 데이터 경로: {data_path}")

        # 데이터 경로 존재 확인
        if not os.path.exists(data_path):
            print(f"❌ 데이터 경로가 존재하지 않습니다: {data_path}")
            # 대체 경로들 시도
            alternative_paths = ["data/", "../data/", "./data/"]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    print(f"✅ 대체 경로를 사용합니다: {data_path}")
                    break
            else:
                raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다. 다음 경로들을 확인해주세요: {[data_path] + alternative_paths}")

        # 파일 경로 설정
        train_file = os.path.join(data_path, 'train.csv')
        val_file = os.path.join(data_path, 'dev.csv')

        print(f"🔍 찾는 파일: {train_file}")

        # 데이터 로드
        train_data = self.preprocessor.make_set_as_df(train_file)
        val_data = self.preprocessor.make_set_as_df(val_file)

        print(f"학습 데이터: {len(train_data)} 샘플")
        print(f"검증 데이터: {len(val_data)} 샘플")

        # 샘플 출력
        print('-' * 100)
        print(f'학습 대화 샘플:\n{train_data["dialogue"][0][:200]}...')
        print(f'학습 요약 샘플:\n{train_data["summary"][0]}')

        # 입력 데이터 생성
        encoder_input_train, decoder_input_train, decoder_output_train = self.preprocessor.make_input(train_data)
        encoder_input_val, decoder_input_val, decoder_output_val = self.preprocessor.make_input(val_data)

        # 토크나이징
        print("토크나이징 중...")

        # 학습 데이터 토크나이징
        tokenized_encoder_inputs = tokenizer(
            encoder_input_train,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            return_token_type_ids=False
        )

        tokenized_decoder_inputs = tokenizer(
            decoder_input_train,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            return_token_type_ids=False
        )

        tokenized_decoder_outputs = tokenizer(
            decoder_output_train,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            return_token_type_ids=False
        )

        # 검증 데이터 토크나이징
        val_tokenized_encoder_inputs = tokenizer(
            encoder_input_val,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            return_token_type_ids=False
        )

        val_tokenized_decoder_inputs = tokenizer(
            decoder_input_val,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            return_token_type_ids=False
        )

        val_tokenized_decoder_outputs = tokenizer(
            decoder_output_val,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            return_token_type_ids=False
        )

        # 데이터셋 생성
        train_dataset = DatasetForTrain(
            tokenized_encoder_inputs,
            tokenized_decoder_inputs,
            tokenized_decoder_outputs,
            len(encoder_input_train)
        )

        val_dataset = DatasetForVal(
            val_tokenized_encoder_inputs,
            val_tokenized_decoder_inputs,
            val_tokenized_decoder_outputs,
            len(encoder_input_val)
        )

        print("데이터셋 생성 완료!")
        return train_dataset, val_dataset

    def prepare_test_dataset(self, tokenizer, use_original=False):
        """테스트용 데이터셋 준비"""
        data_path = self.config['general']['data_path']

        if use_original:
            # 인퍼런스는 항상 원본 test.csv 사용
            test_file = os.path.join('data', 'test.csv')  # 원본 데이터 경로 고정
            print(f"🔍 인퍼런스용 원본 테스트 파일 사용: {test_file}")
        else:
            # 학습용은 기존 로직 유지
            test_file = os.path.join(data_path, 'test_advanced.csv')

            if not os.path.exists(test_file):
                test_file = os.path.join(data_path, 'test.csv')
                print(f"고급 전처리 파일이 없어 원본 파일을 사용합니다: {test_file}")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"테스트 파일을 찾을 수 없습니다: {test_file}")

        test_data = self.preprocessor.make_set_as_df(test_file, is_train=False)
        test_id = test_data['fname']

        print(f"📊 테스트 데이터: {len(test_data)} 샘플")
        print('-' * 100)
        print(f'테스트 대화 샘플:\n{test_data["dialogue"][0][:200]}...')

        encoder_input_test, decoder_input_test = self.preprocessor.make_input(test_data, is_test=True)

        # 토크나이징
        test_tokenized_encoder_inputs = tokenizer(
            encoder_input_test,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            return_token_type_ids=False
        )

        test_dataset = DatasetForInference(
            test_tokenized_encoder_inputs,
            test_id,
            len(encoder_input_test)
        )

        return test_data, test_dataset

    def prepare_fold_dataset(self, tokenizer, train_path: str, val_path: str):
        """K-Fold용 데이터셋 준비"""
        import json

        print(f"📁 K-Fold 데이터 로드: train={train_path}, val={val_path}")

        # JSON 데이터 로드
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        with open(val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)

        print(f"학습 데이터: {len(train_data)} 샘플")
        print(f"검증 데이터: {len(val_data)} 샘플")

        # JSON 데이터를 DataFrame 형태로 변환
        import pandas as pd

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)

        # 필요한 컬럼만 선택 (다양한 JSON 구조 지원)
        def normalize_dataframe(df, df_name):
            """DataFrame을 표준 형식으로 변환"""
            if 'dialogue' in df.columns and 'summary' in df.columns:
                # 표준 형식
                return df[['dialogue', 'summary']].copy()
            elif 'input' in df.columns and 'output' in df.columns:
                # input/output 형식
                normalized_df = df.rename(columns={'input': 'dialogue', 'output': 'summary'})
                return normalized_df[['dialogue', 'summary']].copy()
            else:
                # 지원하지 않는 형식
                available_cols = df.columns.tolist()
                raise ValueError(f"{df_name} 데이터의 형식을 지원하지 않습니다. "
                               f"사용 가능한 컬럼: {available_cols}. "
                               f"필요한 컬럼: ['dialogue', 'summary'] 또는 ['input', 'output']")

        train_df = normalize_dataframe(train_df, "학습")
        val_df = normalize_dataframe(val_df, "검증")

        # 데이터 타입 확인 및 문자열 변환
        train_df['dialogue'] = train_df['dialogue'].astype(str)
        train_df['summary'] = train_df['summary'].astype(str)
        val_df['dialogue'] = val_df['dialogue'].astype(str)
        val_df['summary'] = val_df['summary'].astype(str)

        # 샘플 출력
        print('-' * 100)
        print(f'K-Fold 학습 대화 샘플:\n{train_df["dialogue"].iloc[0][:200]}...')
        print(f'K-Fold 학습 요약 샘플:\n{train_df["summary"].iloc[0]}')

        # 입력 데이터 생성
        encoder_input_train, decoder_input_train, decoder_output_train = self.preprocessor.make_input(train_df)
        encoder_input_val, decoder_input_val, decoder_output_val = self.preprocessor.make_input(val_df)

        # 토크나이징
        print("K-Fold 데이터 토크나이징 중...")

        # 학습 데이터 토크나이징
        encoder_input_train_tokenized = tokenizer(
            encoder_input_train,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            add_special_tokens=True
        )

        decoder_input_train_tokenized = tokenizer(
            decoder_input_train,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            add_special_tokens=False
        )

        decoder_output_train_tokenized = tokenizer(
            decoder_output_train,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            add_special_tokens=False
        )

        # 검증 데이터 토크나이징
        encoder_input_val_tokenized = tokenizer(
            encoder_input_val,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['encoder_max_len'],
            add_special_tokens=True
        )

        decoder_input_val_tokenized = tokenizer(
            decoder_input_val,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            add_special_tokens=False
        )

        decoder_output_val_tokenized = tokenizer(
            decoder_output_val,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config['tokenizer']['decoder_max_len'],
            add_special_tokens=False
        )

        # 데이터셋 생성
        train_dataset = DatasetForTrain(
            encoder_input_train_tokenized,
            decoder_input_train_tokenized,
            decoder_output_train_tokenized,
            len(encoder_input_train)
        )

        val_dataset = DatasetForVal(
            encoder_input_val_tokenized,
            decoder_input_val_tokenized,
            decoder_output_val_tokenized,
            len(encoder_input_val)
        )

        print(f"✅ K-Fold 데이터셋 준비 완료: train={len(train_dataset)}, val={len(val_dataset)}")

        return train_dataset, val_dataset

if __name__ == "__main__":
    # 테스트
    from config_manager import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.load_config()

    processor = DataProcessor(config)
    print("데이터 프로세서 테스트 완료")
