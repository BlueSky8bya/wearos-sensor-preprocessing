"""PPG 전처리 파이프라인(필수 단계 중심)
원본: 전처리(ppg green).ipynb

포함 범위:
- 로딩/컬럼 정리
- 타임스탬프 처리(정렬/중복 제거/세그먼트 분할/윈도우링 등: 원본 함수 내부 구현)
- 필터링/피크 기반 RR 산출 및 SQI 계산(원본 구현을 활용)

주의:
- 본 파일은 "전처리"에 해당하는 핵심 함수만 보관합니다.
- 주차별 지표/통계 비교/시각화 등은 제외했습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .filtering import bandpass_filter
from .sqi import calculate_SQI

def preprocess_ppg_data(df):
    """
    PPG 데이터를 전처리하여 필요한 컬럼만 남깁니다.
    - time/timestamp과 data/value 컬럼을 표준화
    - 두 가지 컬럼명 패턴을 모두 지원: (time, data) 또는 (timestamp, value)
    """
    df_clean = df.copy()

    # 1. 시간 컬럼 확인 및 표준화 (time 또는 timestamp)
    time_col = None
    if 'time' in df_clean.columns:
        time_col = 'time'
    elif 'timestamp' in df_clean.columns:
        time_col = 'timestamp'
    else:
        raise ValueError("시간 컬럼('time' 또는 'timestamp')이 존재하지 않습니다.")

    # 2. 데이터 컬럼 확인 및 표준화 (data 또는 value)
    data_col = None
    if 'data' in df_clean.columns:
        data_col = 'data'
    elif 'value' in df_clean.columns:
        data_col = 'value'
    else:
        raise ValueError("데이터 컬럼('data' 또는 'value')이 존재하지 않습니다.")

    print(f"    사용할 컬럼: 시간={time_col}, 데이터={data_col}")

    # 3. 컬럼명 표준화 (time, data로 통일)
    if time_col != 'time':
        df_clean['time'] = df_clean[time_col]
    if data_col != 'data':
        df_clean['data'] = df_clean[data_col]

    # 4. time과 data가 모두 있는 행만 유지
    before_dropna = len(df_clean)
    df_clean = df_clean.dropna(subset=['time', 'data'])
    after_dropna = len(df_clean)
    print(f"    결측값 제거: {before_dropna} → {after_dropna} ({before_dropna - after_dropna} 행 제거)")

    # 5. 중복된 시간 제거
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['time'])
    after_dedup = len(df_clean)
    print(f"    중복 시간 제거: {before_dedup} → {after_dedup} ({before_dedup - after_dedup} 행 제거)")

    return df_clean
