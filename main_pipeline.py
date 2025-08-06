"""
대화 요약 모델 학습/추론 파이프라인
모듈화된 구조로 각 단계를 독립적으로 실행 가능
"""

import argparse
import sys
import os
from datetime import datetime

# 파이프라인 모듈 import
from pipeline import (
    ConfigManager,
    create_optimized_config_for_rtx3060,
    DataProcessor,
    TrainingManager,
    ModelManager,
    InferenceManager,
    InteractiveInference,
    KFoldManager
)
from pipeline.config_manager import (
    create_high_quality_config_for_rtx3060,
    create_balanced_config_for_rtx3060,
    create_kfold_config_for_rtx3060,
    create_kfold_config_for_high_performance,
    create_fast_kfold_config_for_rtx3060,
    create_rtx3090_baseline_config,
    create_rtx3090_baseline_kfold_config,
    create_rtx3090_baseline_fast_config
)


def generate_model_output_path(config):
    """모델 출력 경로 생성 (모델명 + 타임스탬프)"""
    # 모델명에서 경로에 사용할 수 없는 문자 제거
    model_name = config['general']['model_name']
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')

    # 현재 시간으로 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 폴더명 생성: 모델명_타임스탬프
    folder_name = f"{safe_model_name}_{timestamp}"

    # 전체 경로 생성
    base_output_dir = config['general'].get('output_dir', './model_output/')
    model_output_path = os.path.join(base_output_dir, folder_name)

    return model_output_path


def setup_config(args):
    """설정 초기화"""
    print("=" * 80)
    print("설정 초기화")
    print("=" * 80)

    config_manager = ConfigManager(args.config)

    if args.create_config:
        if args.kfold_rtx3060:
            print(
                f"RTX 3060 K-Fold 교차 검증 설정 생성 중... ({args.kfold_splits} folds)")
            config = create_kfold_config_for_rtx3060(
                n_splits=args.kfold_splits,
                ensemble_method=args.ensemble_method
            )
        elif args.kfold_high_performance:
            print(
                f"고성능 GPU K-Fold 교차 검증 설정 생성 중... ({args.kfold_splits} folds)")
            config = create_kfold_config_for_high_performance(
                n_splits=args.kfold_splits,
                ensemble_method=args.ensemble_method
            )
        elif args.fast_kfold_rtx3060:
            print(f"RTX 3060 빠른 K-Fold 설정 생성 중... ({args.kfold_splits} folds)")
            config = create_fast_kfold_config_for_rtx3060(
                n_splits=args.kfold_splits,
                ensemble_method=args.ensemble_method
            )
        elif args.high_quality_rtx3060:
            print("RTX 3060 고품질 설정 생성 중... (Final Score 최대화)")
            config = create_high_quality_config_for_rtx3060()
        elif args.balanced_rtx3060:
            print("RTX 3060 균형 설정 생성 중... (속도와 품질 균형)")
            config = create_balanced_config_for_rtx3060()
        elif args.rtx3090_baseline:
            print("RTX 3090 Baseline 설정 생성 중... (baseline.ipynb 기반)")
            config = create_rtx3090_baseline_config()
        elif args.rtx3090_baseline_kfold:
            print(f"RTX 3090 Baseline + K-Fold 설정 생성 중... ({args.kfold_splits} folds)")
            config = create_rtx3090_baseline_kfold_config(
                n_splits=args.kfold_splits,
                ensemble_method=args.ensemble_method
            )
        elif args.rtx3090_baseline_fast:
            print("RTX 3090 빠른 Baseline 설정 생성 중... (실험용)")
            config = create_rtx3090_baseline_fast_config()
        elif args.optimize_rtx3060:
            print("RTX 3060 기본 최적화 설정 생성 중...")
            config = create_optimized_config_for_rtx3060()
        else:
            print("기본 설정 생성 중...")
            config_data = config_manager.create_default_config(
                model_name=args.model_name or "digit82/kobart-summarization",
                data_path=args.data_path or "../advanced_processed_data/"
            )
            config_manager.save_config(config_data)
            config = config_manager.load_config()
    else:
        config = config_manager.load_config()

    # 명령행 인자로 설정 오버라이드
    if args.model_name:
        config['general']['model_name'] = args.model_name
    if args.data_path:
        config['general']['data_path'] = args.data_path
        # 증강 데이터 폴더인지 자동 감지
        if os.path.basename(args.data_path).startswith('augmented_'):
            print(f"🔍 증강 데이터 폴더 감지: {os.path.basename(args.data_path)}")
    if args.output_dir:
        config['general']['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_train_epochs'] = args.epochs
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
        config['training']['per_device_eval_batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # 학습 모드인 경우 타임스탬프가 포함된 출력 경로 생성
    if args.mode == 'train':
        model_output_path = generate_model_output_path(config)
        config['general']['output_dir'] = model_output_path
        config['training']['output_dir'] = model_output_path
        config['training']['logging_dir'] = os.path.join(
            model_output_path, 'logs')
        config['inference']['ckt_path'] = model_output_path

        print(f"\n📁 모델 저장 경로: {model_output_path}")

        # 디렉토리 생성
        os.makedirs(model_output_path, exist_ok=True)
        os.makedirs(os.path.join(model_output_path, 'logs'), exist_ok=True)

    config_manager.print_config()
    return config


def save_training_info(config):
    """학습 정보를 파일로 저장"""
    output_dir = config['general']['output_dir']

    # 학습 정보 수집
    training_info = {
        'model_name': config['general']['model_name'],
        'data_path': config['general']['data_path'],
        'training_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_config': config['training'],
        'tokenizer_config': config['tokenizer'],
        'hardware_info': config['general'].get('hardware', 'unknown'),
        'lora_config': config.get('lora', {}),
    }

    # JSON 파일로 저장
    import json
    info_file = os.path.join(output_dir, 'training_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    # README 파일 생성
    readme_content = f"""# 모델 학습 정보

## 기본 정보
- **모델명**: {training_info['model_name']}
- **데이터 경로**: {training_info['data_path']}
- **학습 시작 시간**: {training_info['training_start_time']}
- **하드웨어**: {training_info['hardware_info']}

## 학습 설정
- **에포크 수**: {training_info['training_config']['num_train_epochs']}
- **배치 크기**: {training_info['training_config']['per_device_train_batch_size']}
- **학습률**: {training_info['training_config']['learning_rate']}
- **시퀀스 길이**: {training_info['tokenizer_config']['encoder_max_len']}/{training_info['tokenizer_config']['decoder_max_len']}

## LoRA 설정
- **활성화**: {training_info['lora_config'].get('enabled', False)}
- **QLoRA**: {training_info['lora_config'].get('use_qlora', False)}
- **Rank**: {training_info['lora_config'].get('r', 'N/A')}

## 파일 구조
- `pytorch_model.bin` 또는 `model.safetensors`: 학습된 모델 가중치
- `config.json`: 모델 설정
- `tokenizer.json`, `tokenizer_config.json`: 토크나이저 설정
- `training_args.bin`: 학습 인자
- `trainer_state.json`: 학습 상태
- `logs/`: 텐서보드 로그
- `training_info.json`: 상세 학습 정보
"""

    readme_file = os.path.join(output_dir, 'README.md')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"📄 학습 정보가 저장되었습니다: {info_file}")
    print(f"📄 README 파일이 생성되었습니다: {readme_file}")


def update_training_completion_info(config, trainer):
    """학습 완료 후 정보 업데이트"""
    output_dir = config['general']['output_dir']

    # 기존 정보 로드
    import json
    info_file = os.path.join(output_dir, 'training_info.json')

    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            training_info = json.load(f)
    except:
        training_info = {}

    # 학습 완료 정보 추가
    training_info.update({
        'training_end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_completed': True,
        'final_global_step': trainer.state.global_step if trainer else 'unknown',
        'best_metric': trainer.state.best_metric if trainer and hasattr(trainer.state, 'best_metric') else 'unknown',
        'total_epochs_completed': trainer.state.epoch if trainer else 'unknown',
    })

    # 업데이트된 정보 저장
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)

    print(f"✅ 학습 완료 정보가 업데이트되었습니다: {info_file}")


def run_training(config):
    """학습 실행"""
    print("=" * 80)
    print("학습 파이프라인 시작")
    print("=" * 80)

    # K-Fold 교차 검증 확인
    if config.get('kfold', {}).get('enabled', False):
        return run_kfold_training(config)

    # 일반 학습
    # 학습 정보 저장
    save_training_info(config)

    # 1. 데이터 처리
    print("\n1. 데이터 처리 단계")
    data_processor = DataProcessor(config)

    # 모델과 토크나이저 로드 (데이터 처리용)
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer(
        for_training=True)

    # 데이터셋 준비
    train_dataset, val_dataset = data_processor.prepare_train_dataset(
        tokenizer)

    # 2. 학습 실행
    print("\n2. 모델 학습 단계")
    training_manager = TrainingManager(config)
    trainer = training_manager.train(train_dataset, val_dataset)

    # 학습 완료 후 정보 업데이트
    update_training_completion_info(config, trainer)

    return trainer


def run_kfold_training(config):
    """K-Fold 교차 검증 학습 실행"""
    print("=" * 80)
    print("K-FOLD 교차 검증 학습 파이프라인 시작")
    print("=" * 80)

    kfold_config = config.get('kfold', {})
    n_splits = kfold_config.get('n_splits', 5)
    ensemble_method = kfold_config.get('ensemble_method', 'voting')

    print(f"🔄 K-Fold 설정:")
    print(f"   - Fold 수: {n_splits}")
    print(f"   - 앙상블 방법: {ensemble_method}")
    print(f"   - 계층화: {kfold_config.get('stratified', False)}")

    # K-Fold 학습 정보 저장
    save_kfold_training_info(config)

    # K-Fold 매니저 생성
    kfold_manager = KFoldManager(config)

    # 데이터 경로 확인
    data_path = config['general']['data_path']

    # CSV 데이터를 JSON으로 변환 (필요한 경우)
    json_data_path = prepare_data_for_kfold(data_path)

    try:
        # K-Fold 교차 검증 학습 실행
        fold_results = kfold_manager.run_kfold_training(json_data_path)

        # 성공한 fold 수 확인
        successful_folds = [r for r in fold_results if r.get(
            'training_completed', False)]

        print(f"\n🎉 K-Fold 교차 검증 학습 완료!")
        print(f"   - 성공한 fold: {len(successful_folds)}/{n_splits}")

        if successful_folds:
            # 앙상블 추론 실행 (옵션)
            if kfold_config.get('use_ensemble_inference', True):
                print(f"\n🎯 앙상블 추론 실행 중...")
                ensemble_result = kfold_manager.ensemble_inference()
                print(
                    f"   - 앙상블 추론 완료: {len(ensemble_result.get('predictions', []))} 개 예측")

        return fold_results

    except Exception as e:
        print(f"\n❌ K-Fold 학습 중 오류 발생: {e}")
        raise


def prepare_data_for_kfold(data_path):
    """K-Fold용 데이터 준비 (train.csv + dev.csv 합치고 증강 데이터 지원)"""
    import pandas as pd
    import json
    import os
    import glob

    print(f"📁 K-Fold 데이터 준비 시작: {data_path}")

    # 이미 JSON 파일인 경우
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        print(f"✅ JSON 파일 직접 사용: {data_path}")
        return data_path

    # 증강 데이터 폴더인지 확인 (augmented_로 시작하는 폴더)
    if os.path.basename(data_path).startswith('augmented_'):
        print(f"🔍 증강 데이터 폴더 감지: {os.path.basename(data_path)}")
        return prepare_augmented_data_for_kfold(data_path)

    # 일반 데이터 폴더 처리
    return prepare_standard_data_for_kfold(data_path)


def prepare_standard_data_for_kfold(data_path):
    """표준 데이터 폴더 처리 (train.csv + dev.csv 합치기)"""
    import pandas as pd
    import json
    import os

    train_csv = os.path.join(data_path, 'train.csv')
    dev_csv = os.path.join(data_path, 'dev.csv')

    # 필수 파일 확인
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {train_csv}")

    print(f"📊 데이터 파일 로드 중...")
    print(f"   - Train: {train_csv}")

    # train.csv 로드
    train_df = pd.read_csv(train_csv)
    print(f"   - Train 샘플 수: {len(train_df)}")

    # dev.csv가 있으면 합치기
    if os.path.exists(dev_csv):
        print(f"   - Dev: {dev_csv}")
        dev_df = pd.read_csv(dev_csv)
        print(f"   - Dev 샘플 수: {len(dev_df)}")

        # train과 dev 합치기
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        print(f"✅ Train + Dev 데이터 합치기 완료: {len(combined_df)} 샘플")
    else:
        print(f"⚠️ dev.csv를 찾을 수 없어 train.csv만 사용합니다.")
        combined_df = train_df

    # JSON 형태로 변환
    json_data = []
    for _, row in combined_df.iterrows():
        json_data.append({
            'dialogue': row['dialogue'],
            'summary': row['summary'],
            'fname': row.get('fname', f'combined_{len(json_data)}')
        })

    # JSON 파일로 저장
    json_path = os.path.join(data_path, 'kfold_combined_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"📄 K-Fold용 통합 데이터 저장 완료: {json_path}")
    print(f"   - 총 샘플 수: {len(json_data)}")
    print(f"   - Train 원본: {len(train_df)} 샘플")
    if os.path.exists(dev_csv):
        print(f"   - Dev 원본: {len(dev_df)} 샘플")

    return json_path


def prepare_augmented_data_for_kfold(data_path):
    """증강 데이터 폴더 처리"""
    import pandas as pd
    import json
    import os
    import glob

    print(f"🔍 증강 데이터 폴더 분석 중: {data_path}")

    # 증강 데이터 폴더에서 CSV 파일들 찾기
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"증강 데이터 폴더에 CSV 파일이 없습니다: {data_path}")

    print(f"📊 발견된 CSV 파일들:")
    all_data = []
    total_samples = 0

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        print(f"   - {filename}")

        try:
            df = pd.read_csv(csv_file)
            print(f"     └ 샘플 수: {len(df)}")

            # 데이터 형식 확인 및 변환
            if 'dialogue' in df.columns and 'summary' in df.columns:
                # 표준 형식
                for _, row in df.iterrows():
                    all_data.append({
                        'dialogue': str(row['dialogue']),
                        'summary': str(row['summary']),
                        'fname': row.get('fname', f'{filename}_{len(all_data)}'),
                        'source_file': filename
                    })
            elif 'input' in df.columns and 'output' in df.columns:
                # 다른 형식
                for _, row in df.iterrows():
                    all_data.append({
                        'dialogue': str(row['input']),
                        'summary': str(row['output']),
                        'fname': row.get('fname', f'{filename}_{len(all_data)}'),
                        'source_file': filename
                    })
            else:
                print(f"     ⚠️ 지원하지 않는 컬럼 형식: {df.columns.tolist()}")
                continue

            total_samples += len(df)

        except Exception as e:
            print(f"     ❌ 파일 로드 실패: {e}")
            continue

    if not all_data:
        raise ValueError(f"증강 데이터 폴더에서 유효한 데이터를 찾을 수 없습니다: {data_path}")

    # JSON 파일로 저장
    json_path = os.path.join(data_path, 'kfold_augmented_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 증강 데이터 통합 완료!")
    print(
        f"   - 총 CSV 파일: {len([f for f in csv_files if f.endswith('.csv')])}")
    print(f"   - 총 샘플 수: {len(all_data)}")
    print(f"   - 저장 경로: {json_path}")

    # 데이터 소스별 통계 출력
    source_stats = {}
    for item in all_data:
        source = item.get('source_file', 'unknown')
        source_stats[source] = source_stats.get(source, 0) + 1

    print(f"📈 데이터 소스별 분포:")
    for source, count in sorted(source_stats.items()):
        print(f"   - {source}: {count} 샘플 ({count/len(all_data)*100:.1f}%)")

    return json_path


def save_kfold_training_info(config):
    """K-Fold 학습 정보 저장"""
    output_dir = config['general']['output_dir']
    kfold_config = config.get('kfold', {})

    # K-Fold 학습 정보 수집
    kfold_info = {
        'training_type': 'kfold_cross_validation',
        'model_name': config['general']['model_name'],
        'data_path': config['general']['data_path'],
        'training_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'kfold_config': kfold_config,
        'training_config': config['training'],
        'tokenizer_config': config['tokenizer'],
        'hardware_info': config['general'].get('hardware', 'unknown'),
        'lora_config': config.get('lora', {}),
    }

    # JSON 파일로 저장
    import json
    info_file = os.path.join(output_dir, 'kfold_training_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(kfold_info, f, ensure_ascii=False, indent=2)

    # K-Fold README 파일 생성
    readme_content = f"""# K-Fold 교차 검증 학습 정보

## 기본 정보
- **학습 방식**: K-Fold 교차 검증
- **모델명**: {kfold_info['model_name']}
- **데이터 경로**: {kfold_info['data_path']}
- **학습 시작 시간**: {kfold_info['training_start_time']}
- **하드웨어**: {kfold_info['hardware_info']}

## K-Fold 설정
- **Fold 수**: {kfold_config.get('n_splits', 5)}
- **계층화**: {kfold_config.get('stratified', False)}
- **앙상블 방법**: {kfold_config.get('ensemble_method', 'voting')}
- **랜덤 시드**: {kfold_config.get('random_state', 42)}

## 학습 설정
- **에포크 수**: {kfold_info['training_config']['num_train_epochs']}
- **배치 크기**: {kfold_info['training_config']['per_device_train_batch_size']}
- **학습률**: {kfold_info['training_config']['learning_rate']}
- **시퀀스 길이**: {kfold_info['tokenizer_config']['encoder_max_len']}/{kfold_info['tokenizer_config']['decoder_max_len']}

## LoRA 설정
- **활성화**: {kfold_info['lora_config'].get('enabled', False)}
- **QLoRA**: {kfold_info['lora_config'].get('use_qlora', False)}
- **Rank**: {kfold_info['lora_config'].get('r', 'N/A')}

## 폴더 구조
- `kfold_results/`: K-Fold 결과 폴더
  - `fold_1/`, `fold_2/`, ...: 각 fold별 모델
  - `ensemble/`: 앙상블 결과
  - `kfold_summary.json`: K-Fold 요약 정보
  - `split_info.json`: 데이터 분할 정보
- `kfold_training_info.json`: 상세 학습 정보
"""

    readme_file = os.path.join(output_dir, 'KFOLD_README.md')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"📄 K-Fold 학습 정보가 저장되었습니다: {info_file}")
    print(f"📄 K-Fold README 파일이 생성되었습니다: {readme_file}")


def find_latest_model_path(base_output_dir='./model_output/'):
    """최신 학습된 모델 경로 찾기"""
    if not os.path.exists(base_output_dir):
        return None

    # 타임스탬프가 포함된 폴더들 찾기
    model_folders = []
    for folder in os.listdir(base_output_dir):
        folder_path = os.path.join(base_output_dir, folder)
        if os.path.isdir(folder_path):
            # 폴더명에서 타임스탬프 추출 시도
            parts = folder.split('_')
            if len(parts) >= 3:  # model_name_YYYYMMDD_HHMMSS 형식
                try:
                    timestamp = f"{parts[-2]}_{parts[-1]}"
                    datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    model_folders.append((folder_path, timestamp))
                except:
                    continue

    if not model_folders:
        return None

    # 타임스탬프 기준으로 정렬하여 최신 폴더 반환
    model_folders.sort(key=lambda x: x[1], reverse=True)
    latest_path = model_folders[0][0]

    print(f"🔍 최신 모델 경로 감지: {os.path.basename(latest_path)}")
    return latest_path


def run_inference(config):
    """추론 실행"""
    print("=" * 80)
    print("추론 파이프라인 시작")
    print("=" * 80)

    # 모델 경로 자동 감지 (설정되지 않은 경우)
    if config['inference']['ckt_path'] == './model_output/':
        latest_model_path = find_latest_model_path()
        if latest_model_path:
            config['inference']['ckt_path'] = latest_model_path
            print(f"📁 자동 감지된 모델 경로 사용: {latest_model_path}")
        else:
            print("⚠️ 학습된 모델을 찾을 수 없습니다. 기본 경로를 사용합니다.")

    # K-Fold 앙상블 추론 확인
    kfold_results_path = os.path.join(
        config['inference']['ckt_path'], 'kfold_results')
    if os.path.exists(kfold_results_path) and config.get('kfold', {}).get('use_ensemble_inference', False):
        return run_kfold_inference(config)

    # 일반 추론
    # 1. 데이터 처리
    print("\n1. 테스트 데이터 처리 단계")
    data_processor = DataProcessor(config)

    # 2. 모델 로드
    print("\n2. 모델 로드 단계")
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer(
        for_training=False)

    # 3. 추론 실행 (원본 테스트 데이터 자동 사용)
    print("\n3. 추론 실행 단계")
    inference_manager = InferenceManager(config)
    results = inference_manager.run_inference(model, tokenizer, data_processor)

    return results


def run_kfold_inference(config):
    """K-Fold 앙상블 추론 실행"""
    print("=" * 80)
    print("K-FOLD 앙상블 추론 파이프라인 시작")
    print("=" * 80)

    # K-Fold 매니저 생성
    kfold_manager = KFoldManager(config)

    try:
        # 앙상블 추론 실행
        ensemble_result = kfold_manager.ensemble_inference()

        print(f"\n🎉 K-Fold 앙상블 추론 완료!")
        print(f"   - 앙상블 방법: {ensemble_result.get('method', 'unknown')}")
        print(f"   - 사용된 모델 수: {ensemble_result.get('fold_count', 0)}")
        print(f"   - 예측 결과 수: {len(ensemble_result.get('predictions', []))}")

        return ensemble_result

    except Exception as e:
        print(f"\n❌ K-Fold 앙상블 추론 중 오류 발생: {e}")
        print("일반 추론으로 대체합니다...")

        # 일반 추론으로 fallback
        config['kfold']['use_ensemble_inference'] = False
        return run_inference(config)


def run_interactive(config):
    """대화형 추론 실행"""
    print("=" * 80)
    print("대화형 추론 모드")
    print("=" * 80)

    # 모델 경로 자동 탐지 또는 사용자 지정
    if config['inference']['ckt_path'] == './model_output/':
        # 가장 최근 모델 찾기
        if os.path.exists("model_output"):
            model_dirs = [
                d
                for d in os.listdir("model_output")
                if os.path.isdir(f"model_output/{d}")
            ]
            if model_dirs:
                latest_dir = sorted(model_dirs)[-1]  # 가장 최근 (알파벳 순으로 정렬)
                model_path = f"model_output/{latest_dir}/final"
                print(f"🔍 가장 최근 모델 사용: {model_path}")
            else:
                print("❌ model_output 폴더에 모델이 없습니다.")
                return
        else:
            print("❌ model_output 폴더가 없습니다. 먼저 학습을 실행하세요.")
            return

    if not os.path.exists(model_path):
        print(f"❌ 모델이 없습니다: {model_path}")
        return

    # 모델 로드
    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer(
        for_training=False)

    # 대화형 추론 시작
    interactive = InteractiveInference(config, model, tokenizer)
    interactive.start_interactive_mode()


def main():
    parser = argparse.ArgumentParser(description='대화 요약 모델 파이프라인')

    # 실행 모드
    parser.add_argument('--mode', choices=['train', 'inference', 'interactive', 'config'],
                        default='train', help='실행 모드')

    # 설정 관련
    parser.add_argument('--config', default='config.yaml', help='설정 파일 경로')
    parser.add_argument('--create-config',
                        action='store_true', help='새 설정 파일 생성')
    parser.add_argument('--optimize-rtx3060',
                        action='store_true', help='RTX 3060 기본 최적화 설정')
    parser.add_argument('--high-quality-rtx3060',
                        action='store_true', help='RTX 3060 고품질 설정 (Final Score 최대화)')
    parser.add_argument('--balanced-rtx3060',
                        action='store_true', help='RTX 3060 균형 설정 (속도와 품질 균형)')

    # RTX 3090 Baseline 설정 (baseline.ipynb 기반)
    parser.add_argument('--rtx3090-baseline',
                        action='store_true', help='RTX 3090 Baseline 설정 (baseline.ipynb 기반)')
    parser.add_argument('--rtx3090-baseline-kfold',
                        action='store_true', help='RTX 3090 Baseline + K-Fold 설정')
    parser.add_argument('--rtx3090-baseline-fast',
                        action='store_true', help='RTX 3090 빠른 Baseline 설정 (실험용)')

    # K-Fold 교차 검증 설정
    parser.add_argument('--kfold-rtx3060',
                        action='store_true', help='RTX 3060용 K-Fold 교차 검증 설정')
    parser.add_argument('--kfold-high-performance',
                        action='store_true', help='고성능 GPU용 K-Fold 교차 검증 설정')
    parser.add_argument('--fast-kfold-rtx3060',
                        action='store_true', help='RTX 3060용 빠른 K-Fold 설정 (실험용)')
    parser.add_argument('--kfold-splits', type=int, default=5,
                        help='K-Fold 분할 수 (기본값: 5)')
    parser.add_argument('--ensemble-method', choices=['voting', 'weighted', 'best'],
                        default='voting', help='앙상블 방법 (기본값: voting)')

    # 모델 및 데이터 설정
    parser.add_argument('--model-name', help='사용할 모델명')
    parser.add_argument(
        '--data-path', help='데이터 경로 (일반 폴더 또는 증강 데이터 폴더 - augmented_로 시작하면 자동 감지)')
    parser.add_argument('--output-dir', help='출력 디렉토리')

    # 학습 하이퍼파라미터
    parser.add_argument('--epochs', type=int, help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, help='배치 크기')
    parser.add_argument('--learning-rate', type=float, help='학습률')

    args = parser.parse_args()

    try:
        # 설정 초기화
        config = setup_config(args)

        # 모드별 실행
        if args.mode == 'config':
            print("설정 파일이 생성/업데이트되었습니다.")

        elif args.mode == 'train':
            trainer = run_training(config)
            print("학습이 완료되었습니다!")
            print(f"📁 모델이 저장된 위치: {config['general']['output_dir']}")

        elif args.mode == 'inference':
            results = run_inference(config)
            print(f"추론이 완료되었습니다! 결과: {len(results)} 개")

        elif args.mode == 'interactive':
            run_interactive(config)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
