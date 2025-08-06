"""
앙상블 기법을 통한 성능 향상
여러 모델의 예측을 결합하여 더 높은 성능 달성
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from collections import Counter

class EnsembleInference:
    def __init__(self, model_paths, weights=None):
        """
        앙상블 인퍼런스 초기화
        
        Args:
            model_paths: 모델 경로 리스트
            weights: 각 모델의 가중치 (None이면 동일 가중치)
        """
        self.model_paths = model_paths
        self.weights = weights or [1.0] * len(model_paths)
        self.models = []
        self.tokenizers = []
        
        # 모델들 로드
        self.load_models()
    
    def load_models(self):
        """모든 모델 로드"""
        
        print("📦 앙상블 모델들 로드 중...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i, model_path in enumerate(self.model_paths):
            try:
                print(f"  {i+1}. {model_path} 로드 중...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                model.to(device)
                model.eval()
                
                self.tokenizers.append(tokenizer)
                self.models.append(model)
                print("     ✅ 성공")
                
            except Exception as e:
                print(f"     ❌ 실패: {e}")
                # 실패한 모델은 제외
                self.weights.pop(i)
        
        print(f"✅ 총 {len(self.models)}개 모델 로드 완료")
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        print(f"📊 모델 가중치: {self.weights}")
    
    def generate_single_prediction(self, model, tokenizer, input_text, generation_params):
        """단일 모델로 예측 생성"""
        
        device = next(model.parameters()).device
        
        # 토큰화
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)
        
        # 디코딩
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 후처리
        summary = summary.strip()
        if summary.startswith('다음 대화를 간결하게 요약하세요:'):
            summary = summary.replace('다음 대화를 간결하게 요약하세요:', '').strip()
        
        return summary
    
    def voting_ensemble(self, predictions):
        """투표 기반 앙상블"""
        
        # 단어 단위로 분할
        all_words = []
        for pred in predictions:
            words = pred.split()
            all_words.extend(words)
        
        # 빈도 기반 투표
        word_counts = Counter(all_words)
        
        # 가장 빈번한 단어들로 요약 재구성
        # 간단한 휴리스틱: 각 예측에서 공통으로 나타나는 단어들 우선
        common_words = []
        for word, count in word_counts.most_common():
            if count >= len(predictions) // 2:  # 절반 이상의 모델에서 사용
                common_words.append(word)
        
        # 원본 예측 중 가장 긴 것을 기준으로 재구성
        longest_pred = max(predictions, key=len)
        
        # 공통 단어들이 포함된 예측 선택
        best_pred = longest_pred
        max_common_count = 0
        
        for pred in predictions:
            common_count = sum(1 for word in pred.split() if word in common_words)
            if common_count > max_common_count:
                max_common_count = common_count
                best_pred = pred
        
        return best_pred
    
    def length_weighted_ensemble(self, predictions):
        """길이 가중 앙상블"""
        
        # 적절한 길이의 예측들에 더 높은 가중치
        target_length = 50  # 목표 길이
        
        weighted_scores = []
        for pred in predictions:
            length_score = 1.0 / (1.0 + abs(len(pred.split()) - target_length) / target_length)
            weighted_scores.append(length_score)
        
        # 가장 높은 점수의 예측 선택
        best_idx = np.argmax(weighted_scores)
        return predictions[best_idx]
    
    def semantic_ensemble(self, predictions):
        """의미적 유사성 기반 앙상블"""
        
        # 간단한 키워드 기반 유사성
        # 실제로는 sentence embedding을 사용하는 것이 좋음
        
        keyword_scores = []
        important_keywords = ['에게', '대해', '합니다', '설명', '요청', '이야기', '제안']
        
        for pred in predictions:
            score = sum(1 for keyword in important_keywords if keyword in pred)
            keyword_scores.append(score)
        
        # 키워드 점수가 높은 예측 선택
        if max(keyword_scores) > 0:
            best_idx = np.argmax(keyword_scores)
            return predictions[best_idx]
        else:
            # 키워드가 없으면 길이 기반 선택
            return self.length_weighted_ensemble(predictions)
    
    def ensemble_predict(self, input_text, method='voting'):
        """앙상블 예측"""
        
        if not self.models:
            raise ValueError("로드된 모델이 없습니다.")
        
        # 생성 파라미터
        generation_params = {
            'max_length': 128,
            'min_length': 15,
            'num_beams': 5,
            'length_penalty': 1.2,
            'repetition_penalty': 1.1,
            'early_stopping': True,
            'do_sample': False,
            'no_repeat_ngram_size': 2,
        }
        
        # 각 모델로 예측 생성
        predictions = []
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                pred = self.generate_single_prediction(
                    model, tokenizer, input_text, generation_params
                )
                predictions.append(pred)
            except Exception as e:
                print(f"⚠️ 모델 {i+1} 예측 실패: {e}")
        
        if not predictions:
            return "요약을 생성할 수 없습니다."
        
        # 앙상블 방법 적용
        if method == 'voting':
            return self.voting_ensemble(predictions)
        elif method == 'length_weighted':
            return self.length_weighted_ensemble(predictions)
        elif method == 'semantic':
            return self.semantic_ensemble(predictions)
        else:
            # 기본: 첫 번째 예측 반환
            return predictions[0]

def create_ensemble_models():
    """앙상블용 다양한 모델 학습 제안"""
    
    print("🎯 앙상블용 모델 생성 전략")
    print("="*50)
    
    strategies = [
        {
            "모델": "Model 1 - KoBART Large",
            "설정": "gogamza/kobart-large-v2 + 높은 학습률",
            "특징": "한국어 특화, 공격적 학습"
        },
        {
            "모델": "Model 2 - BART Base",
            "설정": "facebook/bart-base + 보수적 학습률",
            "특징": "안정적 성능, 긴 학습"
        },
        {
            "모델": "Model 3 - T5 Base", 
            "설정": "t5-base + 다른 전처리",
            "특징": "다른 아키텍처, 다양성 확보"
        }
    ]
    
    for strategy in strategies:
        print(f"\n📦 {strategy['모델']}")
        print(f"  설정: {strategy['설정']}")
        print(f"  특징: {strategy['특징']}")
    
    print("\n💡 앙상블 효과:")
    print("  - 단일 모델 대비 3-8점 향상 예상")
    print("  - 안정성 증가")
    print("  - 다양한 관점의 요약 생성")

def discover_trained_models():
    """model_output 폴더에서 학습된 모델들 자동 탐지"""
    
    print("🔍 학습된 모델 탐지 중...")
    
    if not os.path.exists('model_output'):
        print("❌ model_output 폴더가 없습니다.")
        return []
    
    model_paths = []
    model_dirs = [d for d in os.listdir('model_output') if os.path.isdir(f'model_output/{d}')]
    
    for model_dir in sorted(model_dirs):
        model_path = f'model_output/{model_dir}/final'
        info_path = f'model_output/{model_dir}/model_info.json'
        
        if os.path.exists(model_path) and os.path.exists(info_path):
            # 모델 정보 로드
            try:
                import json
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                
                model_paths.append({
                    'path': model_path,
                    'info': model_info,
                    'timestamp': model_info.get('timestamp', ''),
                    'model_name': model_info.get('model_short', 'unknown')
                })
                
                print(f"✅ 발견된 모델: {model_dir}")
                print(f"   - 모델: {model_info.get('model_short', 'unknown')}")
                print(f"   - 시간: {model_info.get('timestamp', 'unknown')}")
                
            except Exception as e:
                print(f"⚠️ 모델 정보 로드 실패: {model_dir} - {e}")
        else:
            print(f"❌ 불완전한 모델: {model_dir}")
    
    return model_paths

def run_ensemble_inference():
    """앙상블 인퍼런스 실행"""
    
    print("🎯 앙상블 인퍼런스 실행")
    print("="*50)
    
    # 학습된 모델들 자동 탐지
    discovered_models = discover_trained_models()
    
    if len(discovered_models) < 2:
        print("⚠️ 앙상블을 위해서는 최소 2개의 모델이 필요합니다.")
        print(f"현재 발견된 모델: {len(discovered_models)}개")
        print("💡 scripts/quick_train_optimized.py를 다른 설정으로 여러 번 실행하세요.")
        return
    
    # 모델 경로만 추출
    existing_models = [model['path'] for model in discovered_models]
    
    print(f"\n📊 앙상블에 사용할 모델: {len(existing_models)}개")
    for i, model in enumerate(discovered_models):
        print(f"  {i+1}. {model['model_name']} ({model['timestamp']})")
    
    # 앙상블 인퍼런스 초기화
    ensemble = EnsembleInference(existing_models)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv('data/test.csv')
    print(f"📊 테스트 데이터: {len(test_df)} 샘플")
    
    # 앙상블 예측 실행
    predictions = []
    
    print("🔄 앙상블 인퍼런스 진행 중...")
    for idx, row in test_df.iterrows():
        # 입력 최적화
        dialogue = row['dialogue']
        dialogue = dialogue.replace('A:', '[화자1]:').replace('B:', '[화자2]:')
        input_text = f"다음 대화를 간결하게 요약하세요: {dialogue}"
        
        # 앙상블 예측
        try:
            pred = ensemble.ensemble_predict(input_text, method='semantic')
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️ 샘플 {idx} 예측 실패: {e}")
            predictions.append("요약을 생성할 수 없습니다.")
        
        if (idx + 1) % 50 == 0:
            print(f"  진행률: {idx + 1}/{len(test_df)} ({(idx + 1)/len(test_df)*100:.1f}%)")
    
    # 결과 저장
    print("💾 앙상블 결과 저장 중...")
    submission = pd.read_csv('data/sample_submission.csv')
    submission['summary'] = predictions
    submission.to_csv('ensemble_submission.csv', index=False)
    
    print("✅ 앙상블 인퍼런스 완료!")
    print("📁 결과 파일: ensemble_submission.csv")
    
    # 샘플 결과 확인
    print("\n📋 앙상블 결과 샘플:")
    for i in range(min(3, len(predictions))):
        print(f"  {i+1}. {predictions[i]}")

if __name__ == "__main__":
    import os
    
    print("🎯 앙상블 기법을 통한 성능 향상")
    print("예상 효과: +3-8점")
    print("="*60)
    
    # 앙상블 모델 생성 전략 안내
    create_ensemble_models()
    
    print("\n" + "="*60)
    
    # 앙상블 인퍼런스 실행
    run_ensemble_inference()