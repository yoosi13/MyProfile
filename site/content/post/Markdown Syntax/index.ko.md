---
title: "LLM LoRA 파인튜닝을 통한 AI 텍스트 판별 모델 학습 및 추론"
date: 2025-05-25
description: "Hugging Face, PEFT(LoRA), BitsAndBytes를 활용한 4-bit 양자화 및 시퀀스 분류 모델 파인튜닝과 추론 파이프라인 구축 과정"
tags:
- python
- llm
- lora
- machine-learning
- transformers
categories:
- AI/ML
image: pawel-czerwinski-8uZPynIu-rQ-unsplash.jpg
---

Hugging Face 라이브러리와 PEFT(LoRA) 기법을 활용하여 대형 언어 모델(LLM)을 파인튜닝하고, 텍스트가 인간이 작성한 것인지 AI가 생성한 것인지 판별하는 이진 분류 모델을 구축 및 추론하는 전 과정을 정리한 글입니다.

<!--more-->

## 1. 환경 설정 및 라이브러리 설치 (Import and Install)

모델 학습과 데이터 처리에 필요한 주요 라이브러리를 임포트하고, 효율적인 파인튜닝을 위한 패키지들을 설치합니다.

- **주요 패키지**: `transformers`, `peft`, `bitsandbytes`, `datasets`, `accelerate` 등
- 4-bit 양자화 및 저비용 고효율 학습을 위한 환경을 준비합니다.

```python
import os
import warnings
import pandas as pd
import torch
import numpy as np

!pip -qq install peft
!pip -qq install bitsandbytes
!pip -qq install GPUtil
!pip -qq install transformers==4.43.1
!pip -qq install transformers accelerate sentencepiece
!pip -qq install accelerate==0.34.2
!pip -qq install torch
!pip -qq install langdetect
```
<br>

## 2. 세부 설정 (Configure)

재현성(Reproducibility)을 확보하기 위해 고정 시드(Seed)를 설정하고, 모델 학습에 사용할 하이퍼파라미터와 경로를 지정합니다.

```python
from transformers import set_seed

SEED = 123
set_seed(SEED)

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_DIR = '/kaggle/input/2025-sw-digital-competition-datasets'
DIR = '/kaggle/working/'
NUM_WORKERS = os.cpu_count()
NUM_CLASSES = 2

EPOCHS, R, LORA_ALPHA, LORA_DROPOUT = 1, 64, 32, 0.3
BATCH_SIZE = 6
MODEL_ID = '/kaggle/input/llama-2/pytorch/7b-hf/1'
```

### 3. 데이터 전처리 및 구성 (Data Processing)

인간이 작성한 텍스트 데이터와 AI가 생성한 텍스트 데이터를 각각 불러와 라벨 기준으로 필터링하고, DatasetDict 형태로 재구성합니다.

- 데이터 로드 및 필터링: Human 데이터(label == 0)와 AI 데이터(label == 1, generated >= 0.6)를 결합
- 데이터셋 셔플: concatenate_datasets와 shuffle을 통해 학습 및 테스트 셋 구성

```python
from datasets import load_dataset, concatenate_datasets, DatasetDict

# 1. 데이터 로드
dataset_53 = load_dataset('csv', data_files='/kaggle/input/digital-competition-no-train-dataset/non_train_human_token.csv')
dataset_ai = load_dataset('csv', data_files='/kaggle/input/digital-competition-no-train-dataset/top20000_ai_like_sentences (1).csv')

# 2. 라벨 기준 필터링 및 컬럼 정리
human_data = dataset_53['train'].filter(lambda x: x['label'] == 0)
ai_data = dataset_ai['train'].filter(lambda x: x['label'] == 1)
ai_data_filtered_by_generated = ai_data.filter(lambda x: x['generated'] >= 0.6)

human_data = human_data.remove_columns(['text_length', 'token_count', 'token_bin'])
ai_data_filtered_by_generated = ai_data_filtered_by_generated.remove_columns(['generated'])

# 3. 데이터 합치기 및 쪼개기
combined_dataset = concatenate_datasets([human_data, ai_data_filtered_by_generated]).shuffle(seed=42)
test_dataset = combined_dataset.select(range(2000))

dataset = DatasetDict({
    'train': combined_dataset,
    'test' : test_dataset
})
```