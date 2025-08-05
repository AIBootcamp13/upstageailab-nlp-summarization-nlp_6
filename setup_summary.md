# 환경 설정 요약

이 문서는 `finetune_summarizer_3090.py` 스크립트 실행을 위해 필요한 라이브러리 및 패키지 설치 과정을 요약합니다.

## 1. 시스템 패키지 (APT)

스크립트 실행 중 `triton` 라이브러리가 C 컴파일러를 필요로 하여 오류가 발생했습니다. 이를 해결하기 위해 `gcc` 컴파일러를 설치했습니다.

### 설치 패키지
- **`gcc`**: C 언어 컴파일러

### 설치 명령어

```bash
# 1. 패키지 목록 업데이트
apt-get update

# 2. gcc 설치
apt-get install -y gcc
```

## 2. Python 라이브러리 (pip)

스크립트의 주요 의존성인 `unsloth` 라이브러리를 실행하는 과정에서 추가적으로 `unsloth_zoo` 패키지가 필요했습니다.

### 설치 패키지
- **`unsloth_zoo`**: `unsloth` 라이브러리의 추가 기능 및 최적화를 위한 의존성 패키지입니다.

### 설치 명령어

```bash
pip3 install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip3 install "datasets" "accelerate" "trl" "transformers" "bitsandbytes"
# unsloth_zoo 설치
pip install unsloth_zoo
```
