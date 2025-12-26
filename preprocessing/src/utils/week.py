"""주차(Week) 라벨 유틸 (선택)
원본: 전처리(ppg green).ipynb / 전처리(response, heart rate, step count).ipynb

전처리 이후, 주차 단위로 데이터를 묶을 때 사용됩니다.
"""
from __future__ import annotations

import pandas as pd

def load_week_cutoffs():
    """response.csv 파일에서 주차별 lastUpdate 정보를 로드"""
    global last_update_map

    response_path = os.path.join(BASE_DIR, UID, "response.csv")
    if os.path.exists(response_path):
        try:
            df_response = pd.read_csv(response_path, encoding="cp949")
            df_response["last_ts"] = pd.to_datetime(df_response["lastUpdate"])

            for wk in sorted(df_response["week"].unique()):
                ts = df_response.loc[df_response["week"] == wk, "last_ts"].iloc[0]
                last_update_map[wk] = ts

            print("=== 주차별 lastUpdate 정보 로드 완료 ===")
            for wk, ts in last_update_map.items():
                print(f"week {wk} lastUpdate: {ts}")
            print("=========================================\n")
        except Exception as e:
            print(f"Warning: response.csv 로드 중 오류 발생: {e}")
    else:
        print(f"Warning: response.csv 파일을 찾을 수 없습니다: {response_path}")

def assign_week_label(ts_kst, cutoffs):
    """
    ts_kst(naive KST timestamp)를, cutoffs 리스트에 정의된 '주차 경계 시점'을 기준으로
    올바른 주차(week) 정수로 반환.
    cutoffs: [(week_index, Timestamp), ...] 형태로 주차 순서대로 정렬되어야 함.
    예를 들어 [(1, T1), (2, T2), (3, T3), ...] 일 때:
      ts < T1  → 0
      T1 ≤ ts < T2 → 1
      T2 ≤ ts < T3 → 2
      ts ≥ T3 → 3  (마지막 주차)
    """
    for wk, cutoff_ts in cutoffs:
        if ts_kst < cutoff_ts:
            return wk - 1
    # 모든 cutoff를 넘으면 가장 마지막 주차
    return cutoffs[-1][0]
