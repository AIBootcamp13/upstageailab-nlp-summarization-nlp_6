"""
프로젝트 정보 및 상태 확인 유틸리티
"""

import os
import pandas as pd
from pathlib import Path

def show_project_structure():
    """프로젝트 구조 출력"""
    
    print("📁 현재 프로젝트 구조:")
    print("├── 🚀 핵심 실행 파일")
    print("│   ├── main.py")
    print("│   ├── main_pipeline.py") 
    print("│   └── config.yaml")
    print("├── 📊 analysis/ (데이터 분석)")
    print("├── 🔧 preprocessing/ (전처리)")
    print("├── 🧪 experiments/ (실험)")
    print("├── 🛠️ utils/ (유틸리티)")
    print("├── 📚 docs/ (문서)")
    print("├── 📦 archive/ (사용안함)")
    print("├── 🏗️ pipeline/ (모듈)")
    print("├── ⚡ scripts/ (빠른실행)")
    print("├── 💾 data/ (원본데이터)")
    print("└── 📋 lenient_processed_data/ (최종전처리)")

def check_data_status():
    """데이터 상태 확인"""
    
    print("\n📊 데이터 현황:")
    
    # 원본 데이터
    if os.path.exists("data"):
        train_df = pd.read_csv("data/train.csv")
        dev_df = pd.read_csv("data/dev.csv") 
        test_df = pd.read_csv("data/test.csv")
        
        print(f"  원본 데이터:")
        print(f"    - Train: {len(train_df)} 샘플")
        print(f"    - Dev: {len(dev_df)} 샘플")
        print(f"    - Test: {len(test_df)} 샘플")
    
    # 전처리된 데이터
    if os.path.exists("lenient_processed_data"):
        files = os.listdir("lenient_processed_data")
        print(f"  전처리된 데이터: {len(files)}개 파일")
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(f"lenient_processed_data/{file}")
                print(f"    - {file}: {len(df)} 샘플")

def show_quick_commands():
    """빠른 실행 명령어들"""
    
    print("\n⚡ 빠른 실행 명령어:")
    print("  📊 데이터 분석:")
    print("    python analysis/data_analysis.py")
    print("  🔧 데이터 전처리:")
    print("    python preprocessing/enhanced_preprocess.py")
    print("  🚀 모델 학습:")
    print("    python scripts/quick_train.py")
    print("  🎯 인퍼런스:")
    print("    python scripts/quick_inference.py")
    print("  🏗️ 파이프라인 실행:")
    print("    python main_pipeline.py")

if __name__ == "__main__":
    show_project_structure()
    check_data_status()
    show_quick_commands()