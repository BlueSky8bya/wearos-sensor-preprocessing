"""설문 전처리(최소) + PHQ-9/GAD-7 라벨 생성
원본: 전처리(response, heart rate, step count).ipynb, surveyAnalysis.ts

포함:
- 설문 원본 로딩/정리(노트북 구현)
- PHQ-9/GAD-7 점수화/이분화(Python 최소 구현)

제외:
- 기타 척도(ERQ/WHOQOL 등) 분석 파트
- 리포트 생성/통계 비교
"""
from __future__ import annotations

import pandas as pd
import numpy as np

def preprocess_response():
    """
    response.csv 파일을 읽어서
    1) 각 주차별 lastUpdate를 last_update_map에 저장하고
    2) PHQ-9, GAD-7 원점수를 계산한 후
    3) (uid, week, phq9_score, gad7_score, phq9_label, gad7_label) 형태로 long format DataFrame 생성
    4) 디버깅용으로 출력하고
    5) CSV로 저장
    """
    path = get_file_path("response.csv")
    df = pd.read_csv(path, encoding="cp949")

    df["last_ts"] = pd.to_datetime(df["lastUpdate"])

    # === 주차별 마지막 업데이트 시각 저장 ===
    for wk in sorted(df["week"].unique()):
        ts = df.loc[df["week"] == wk, "last_ts"].iloc[0]
        last_update_map[wk] = ts

    # === PHQ-9 점수 계산 ===
    phq_item_cols = []
    for i in range(9):
        answer_cols = [f"PHQ-9[{i}].answers[{j}]" for j in range(4)]
        df[f"PHQ9_item_{i}"] = (
            df[answer_cols].fillna(False)
            .apply(lambda row: sum(int(row[col]) * j for j, col in enumerate(answer_cols)), axis=1)
        )
        phq_item_cols.append(f"PHQ9_item_{i}")
    df["phq9_score"] = df[phq_item_cols].sum(axis=1)

    # === GAD-7 점수 계산 ===
    gad_item_cols = []
    for i in range(7):
        answer_cols = [f"GAD-7[{i}].answers[{j}]" for j in range(4)]
        df[f"GAD7_item_{i}"] = (
            df[answer_cols].fillna(False)
            .apply(lambda row: sum(int(row[col]) * j for j, col in enumerate(answer_cols)), axis=1)
        )
        gad_item_cols.append(f"GAD7_item_{i}")
    df["gad7_score"] = df[gad_item_cols].sum(axis=1)

    # === 라벨링 추가 ===
    df["PHQ_label"] = pd.cut(
        df["phq9_score"],
        bins=[-1, 4, 9, 19, 27],
        labels=[0, 1, 2, 3],
        right=True
    ).astype(int)

    df["GAD_label"] = (df["gad7_score"] >= 6).astype(int)

    # === long format 생성 ===
    df_long = df[["uid", "week", "phq9_score", "gad7_score", "PHQ_label", "GAD_label"]].copy()
    df_long = df_long.rename(columns={
        "phq9_score": "PHQ_9",
        "gad7_score": "GAD_7"
    })

    # === 디버깅 출력 ===
    print("===== 주차별 PHQ-9 / GAD-7 Raw 점수 및 라벨 (long format, 디버깅용) =====")
    print(df_long)
    print("=====================================================================\n")

    print("===== 주차별 lastUpdate (디버깅용) =====")
    for wk, ts in last_update_map.items():
        print(f"week {wk} lastUpdate: {ts}")
    print("========================================\n")

    # === CSV 저장 ===
    output_path = os.path.join(OUTPUT_DIR, "response_processed.csv")
    df_long.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Processed (PHQ-9/GAD-7 + 라벨링 포함) 파일 저장 완료: {output_path}")


"""PHQ-9 / GAD-7 점수화 (전처리 - 라벨 생성)

- TS(surveyAnalysis.ts)에도 동일/확장 로직이 있으나, 분석 파이프라인에서 바로 쓰기 위한 Python 버전입니다.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


def score_phq9(items: List[Optional[int]]) -> Dict[str, object]:
    x = [(0 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)) for v in items]
    total = int(sum(x))
    core1, core2 = x[0], x[1]
    core_pos = (core1 >= 2) or (core2 >= 2)

    if total <= 4:
        sev = "우울 아님"
    elif total <= 9:
        sev = "가벼운 증상"
    elif total <= 14:
        sev = "경미한 증상"
    elif total <= 19:
        sev = "중한 증상"
    else:
        sev = "심한 증상"

    sev_screened = sev if core_pos else "우울 아님"
    binary = 1 if (core_pos and total >= 10) else 0

    return {
        "phq9_total": total,
        "phq9_severity": sev_screened,
        "phq9_binary": binary,
        "phq9_core_positive": int(core_pos),
    }


def score_gad7(items: List[Optional[int]]) -> Dict[str, object]:
    x = [(0 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)) for v in items]
    total = int(sum(x))
    binary = 1 if total >= 6 else 0
    return {"gad7_total": total, "gad7_binary": binary}

