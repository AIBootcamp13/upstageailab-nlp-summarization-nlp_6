import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class SummarizationDataset(Dataset):
    """
    Hugging Face Trainer와 함께 사용될 PyTorch Dataset 클래스입니다.
    """
    def __init__(self, tokenized_data):
        self.data = tokenized_data
        self.data_len = len(self.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return item

    def __len__(self):
        return self.data_len

class DataPreprocessor:
    """
    데이터 로딩, 전처리, 토큰화 등 데이터 관련 모든 작업을 담당하는 클래스입니다.
    """
    def __init__(self, config: dict, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # Causal LM과 Seq2Seq 모델 모두를 위해 토큰을 명시적으로 가져옵니다.
        self.bos_token = tokenizer.bos_token or "<s>"
        self.eos_token = tokenizer.eos_token or "</s>"

    def _build_causal_lm_prompt(self, dialogue, summary=None):
        """Causal LM을 위한 프롬프트를 생성합니다."""
        prompt = f"### 대화:\n{dialogue}\n\n### 요약:\n"
        if summary:
            prompt += f"{summary}{self.eos_token}"
        return prompt

    def _load_and_prepare(self, data_path: str):
        df = pd.read_csv(data_path)
        df.dropna(subset=['dialogue', 'summary'], inplace=True)

        model_type = self.config['model']['type']
        is_causal_lm = 'koalpaca' in model_type

        if is_causal_lm:
            print("Processing data for Causal LM...")
            full_prompts = [self._build_causal_lm_prompt(d, s) for d, s in zip(df['dialogue'], df['summary'])]
            return full_prompts, None, None
        else:
            print("Processing data for Seq2Seq model...")
            prefix = self.config['model'].get('prefix', '')
            encoder_input = (prefix + df['dialogue']).tolist()
            decoder_input = (self.bos_token + df['summary']).tolist()
            decoder_output = (df['summary'] + self.eos_token).tolist()
            return encoder_input, decoder_input, decoder_output

    def tokenize_data(self, encoder_input, decoder_input, decoder_output):
        model_type = self.config['model']['type']
        is_causal_lm = 'koalpaca' in model_type

        if is_causal_lm:
            tokenized_data = self.tokenizer(
                encoder_input,
                max_length=self.config['model']['encoder_max_len'],
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            tokenized_data['labels'] = tokenized_data['input_ids'].clone()
            return tokenized_data
        else:
            tokenized_encoder = self.tokenizer(
                encoder_input,
                max_length=self.config['model']['encoder_max_len'],
                truncation=True,
                padding='max_length',
            )
            tokenized_decoder_input = self.tokenizer(
                decoder_input,
                max_length=self.config['model']['decoder_max_len'],
                truncation=True,
                padding='max_length',
            )
            tokenized_decoder_output = self.tokenizer(
                decoder_output,
                max_length=self.config['model']['decoder_max_len'],
                truncation=True,
                padding='max_length',
            )
            labels = torch.tensor(tokenized_decoder_output['input_ids'])
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                'input_ids': torch.tensor(tokenized_encoder.input_ids),
                'attention_mask': torch.tensor(tokenized_encoder.attention_mask),
                'decoder_input_ids': torch.tensor(tokenized_decoder_input.input_ids),
                'decoder_attention_mask': torch.tensor(tokenized_decoder_input.attention_mask),
                'labels': labels
            }

    def setup_datasets(self):
        """
        훈련 및 검증 데이터셋을 설정하고, config 설정에 따라 캐시를 관리합니다.
        """
        print("Starting setup_datasets...")
        
        use_cache = self.config.get('use_data_cache', True)
        
        train_path = os.path.join(self.config['data_dir'], 'train.csv')
        val_path = os.path.join(self.config['data_dir'], 'dev.csv')
        
        model_name_for_path = self.config['model']['name'].replace("/", "_")
        cache_dir = os.path.join(self.config['data_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        train_cache_path = os.path.join(cache_dir, f"train_{model_name_for_path}.pt")
        val_cache_path = os.path.join(cache_dir, f"val_{model_name_for_path}.pt")

        if use_cache and os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
            print(f"Cache is enabled. Loading tokenized data from cache...")
            tokenized_train = torch.load(train_cache_path, weights_only=False)
            tokenized_val = torch.load(val_cache_path, weights_only=False)
        else:
            if not use_cache:
                print("Cache is disabled. Tokenizing data directly...")
            else:
                print(f"Cache is enabled but not found. Tokenizing data...")
                
            enc_train, dec_in_train, dec_out_train = self._load_and_prepare(train_path)
            tokenized_train = self.tokenize_data(enc_train, dec_in_train, dec_out_train)

            enc_val, dec_in_val, dec_out_val = self._load_and_prepare(val_path)
            tokenized_val = self.tokenize_data(enc_val, dec_in_val, dec_out_val)
            
            if use_cache:
                print("Saving tokenized data to cache...")
                torch.save(tokenized_train, train_cache_path)
                torch.save(tokenized_val, val_cache_path)
                print(f"Tokenized data saved to {cache_dir}")

        train_dataset = SummarizationDataset(tokenized_train)
        # ✨ [핵심 수정] val_dataset을 tokenized_val로 올바르게 초기화합니다.
        val_dataset = SummarizationDataset(tokenized_val)

        print("Finished setup_datasets.")
        return train_dataset, val_dataset

    def prepare_for_inference(self, data_path: str):
        """
        추론(테스트) 데이터를 준비합니다.
        """
        print(f"Loading inference data from {data_path}...")
        df = pd.read_csv(data_path)
        df.dropna(subset=['dialogue'], inplace=True)
        
        fnames = df['fname'].tolist()
        
        model_type = self.config['model']['type']
        is_causal_lm = 'koalpaca' in model_type

        if is_causal_lm:
            encoder_input = [self._build_causal_lm_prompt(dialogue) for dialogue in df['dialogue']]
        else:
            prefix = self.config['model'].get('prefix', '')
            encoder_input = (prefix + df['dialogue']).tolist()

        tokenized_data = self.tokenizer(
            encoder_input,
            max_length=self.config['model']['encoder_max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        if len(fnames) != len(tokenized_data['input_ids']):
            raise RuntimeError("Mismatch between number of fnames and tokenized inputs after processing.")

        return tokenized_data, fnames
