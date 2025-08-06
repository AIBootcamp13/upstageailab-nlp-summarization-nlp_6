"""
추론 관리 모듈
"""

import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

class InferenceManager:
    """추론 관리 클래스"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run_inference(self, model, tokenizer, data_processor):
        """추론 실행 (원본 테스트 데이터 사용)"""
        print("=" * 80)
        print("모델 추론 시작 (원본 테스트 데이터 사용)")
        print("=" * 80)
        
        model.eval()
        
        # 원본 테스트 데이터 준비
        test_data, test_dataset = data_processor.prepare_test_dataset(tokenizer, use_original=True)
        
        # 데이터로더 생성
        dataloader = DataLoader(
            test_dataset, 
            batch_size=self.config['inference']['batch_size'],
            shuffle=False
        )
        
        summaries = []
        text_ids = []
        
        print(f"총 {len(dataloader)} 배치 처리 중...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="추론 진행"):
                # ID 저장
                text_ids.extend(batch['ID'])
                
                # 생성
                generated_ids = model.generate(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    no_repeat_ngram_size=self.config['inference']['no_repeat_ngram_size'],
                    early_stopping=self.config['inference']['early_stopping'],
                    max_length=self.config['inference']['generate_max_length'],
                    num_beams=self.config['inference']['num_beams'],
                )
                
                # 디코딩
                for ids in generated_ids:
                    result = tokenizer.decode(ids, skip_special_tokens=False)
                    summaries.append(result)
        
        # 후처리
        cleaned_summaries = self._clean_summaries(summaries)
        
        # 결과 저장 (이제 누락 샘플 처리가 불필요함)
        output_df = self._save_results_simple(test_data, cleaned_summaries)
        
        print("=" * 80)
        print("모델 추론 완료!")
        print(f"결과가 {self.config['inference']['result_path']}에 저장되었습니다.")
        print("=" * 80)
        
        return output_df
    
    def _clean_summaries(self, summaries):
        """요약문 후처리"""
        remove_tokens = self.config['inference']['remove_tokens']
        cleaned_summaries = summaries.copy()
        
        for token in remove_tokens:
            cleaned_summaries = [
                sentence.replace(token, " ").strip() 
                for sentence in cleaned_summaries
            ]
        
        return cleaned_summaries
    
    def _save_results(self, test_data, summaries):
        """결과 저장 (누락된 샘플 자동 처리 포함)"""
        output_df = pd.DataFrame({
            "fname": test_data['fname'],
            "summary": summaries,
        })
        
        # sample_submission과 비교하여 누락된 샘플 처리
        sample_submission_path = "data/sample_submission.csv"
        if os.path.exists(sample_submission_path):
            try:
                sample_df = pd.read_csv(sample_submission_path)
                required_fnames = sample_df['fname'].tolist()
                existing_fnames = output_df['fname'].tolist()
                missing_fnames = [fname for fname in required_fnames if fname not in existing_fnames]
                
                if missing_fnames:
                    print(f"⚠️ 누락된 샘플 {len(missing_fnames)}개 발견, 기본 요약으로 추가합니다.")
                    
                    # 누락된 샘플들을 기본 요약으로 추가
                    missing_data = []
                    for fname in missing_fnames:
                        missing_data.append({
                            'fname': fname,
                            'summary': '대화 내용을 요약한 결과입니다.'
                        })
                    
                    missing_df = pd.DataFrame(missing_data)
                    output_df = pd.concat([output_df, missing_df], ignore_index=True)
                    
                    # sample_submission 순서에 맞춰 정렬
                    output_df = output_df.set_index('fname').loc[required_fnames].reset_index()
                    
                    print(f"✅ 총 {len(output_df)}개 샘플로 완성되었습니다.")
                
            except Exception as e:
                print(f"⚠️ sample_submission 처리 중 오류: {e}")
        
        # 결과 디렉토리 생성
        result_path = self.config['inference']['result_path']
        os.makedirs(result_path, exist_ok=True)
        
        # CSV 저장
        output_file = os.path.join(result_path, "output.csv")
        output_df.to_csv(output_file, index=False)
        
        print(f"결과 파일 저장: {output_file}")
        
        # 샘플 출력
        print("\n=== 추론 결과 샘플 ===")
        for i in range(min(3, len(output_df))):
            print(f"\n[샘플 {i+1}]")
            print(f"파일명: {output_df.iloc[i]['fname']}")
            print(f"요약: {output_df.iloc[i]['summary']}")
        
        return output_df
    
    def _save_results_simple(self, test_data, summaries):
        """간단한 결과 저장 (원본 데이터 사용시 누락 샘플 없음)"""
        output_df = pd.DataFrame({
            "fname": test_data['fname'],
            "summary": summaries,
        })
        
        # 결과 디렉토리 생성
        result_path = self.config['inference']['result_path']
        os.makedirs(result_path, exist_ok=True)
        
        # CSV 저장
        output_file = os.path.join(result_path, "output.csv")
        output_df.to_csv(output_file, index=False)
        
        print(f"✅ 결과 파일 저장: {output_file}")
        print(f"📊 총 샘플 수: {len(output_df)}")
        
        # 샘플 출력
        print("\n=== 추론 결과 샘플 ===")
        for i in range(min(3, len(output_df))):
            print(f"\n[샘플 {i+1}]")
            print(f"파일명: {output_df.iloc[i]['fname']}")
            print(f"요약: {output_df.iloc[i]['summary']}")
        
        return output_df
    
    def evaluate_with_reference(self, predictions, references):
        """참조 요약과 비교 평가"""
        from rouge import Rouge
        
        rouge = Rouge()
        
        try:
            scores = rouge.get_scores(predictions, references, avg=True)
            
            print("\n=== 평가 결과 ===")
            for metric, values in scores.items():
                print(f"{metric.upper()}:")
                print(f"  Precision: {values['p']:.4f}")
                print(f"  Recall: {values['r']:.4f}")
                print(f"  F1-Score: {values['f']:.4f}")
            
            return scores
            
        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            return None
    
    def generate_single_summary(self, model, tokenizer, dialogue):
        """단일 대화에 대한 요약 생성"""
        model.eval()
        
        # 토크나이징
        inputs = tokenizer(
            dialogue,
            return_tensors='pt',
            max_length=self.config['tokenizer']['encoder_max_len'],
            truncation=True,
            padding=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                no_repeat_ngram_size=self.config['inference']['no_repeat_ngram_size'],
                early_stopping=self.config['inference']['early_stopping'],
                max_length=self.config['inference']['generate_max_length'],
                num_beams=self.config['inference']['num_beams'],
            )
        
        # 디코딩
        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 후처리
        for token in self.config['inference']['remove_tokens']:
            summary = summary.replace(token, " ")
        
        return summary.strip()

class InteractiveInference:
    """대화형 추론 클래스"""
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.inference_manager = InferenceManager(config)
    
    def start_interactive_mode(self):
        """대화형 모드 시작"""
        print("=" * 80)
        print("대화형 요약 생성 모드")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("=" * 80)
        
        while True:
            try:
                # 사용자 입력
                dialogue = input("\n대화를 입력하세요:\n> ")
                
                if dialogue.lower() in ['quit', 'exit', '종료']:
                    print("대화형 모드를 종료합니다.")
                    break
                
                if not dialogue.strip():
                    print("대화를 입력해주세요.")
                    continue
                
                # 요약 생성
                print("\n요약 생성 중...")
                summary = self.inference_manager.generate_single_summary(
                    self.model, self.tokenizer, dialogue
                )
                
                print(f"\n생성된 요약:\n{summary}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n대화형 모드를 종료합니다.")
                break
            except Exception as e:
                print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 테스트
    from config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    inference_manager = InferenceManager(config)
    print("추론 매니저 테스트 완료")