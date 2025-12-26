"""심박(Heart Rate) 전처리 (SOP 전처리 범위)
원본: 전처리(response, heart rate, step count).ipynb

포함 범위:
- 타임스탬프 정규화/정렬/중복 처리
- 1분 단위 집계(ts_kst_min, HR_mean)

제외:
- 주차/시간대 패턴 feature 계산(분석 단계)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

def preprocess_watch_heart_rate():
    """
    watch_heart_rate.csv 파일을 전처리하여, 주차별 다양한 심박 특성(feature)을 추출하고
    'watch_heart_rate_processed.csv'로 저장합니다.

    주요 단계:
      1) 원본 CSV 읽기
      2) timestamp UTC → KST(naive) 변환
      3) 주차별 경계 시점(cutoffs) 설정 및 week_label 부여
      4) 1분 단위 평균 심박수(HR_mean) 계산
      5) 분당 평균 HR에 대해 다시 week_label 부여
      6) 주차별로 다음 피처 계산 (변경된 부분만 발췌).
      7) (uid, week)별로 피처를 하나의 DataFrame에 모아서 저장
    """
    # 1) CSV 읽기
    path = os.path.join(BASE_DIR, UID, "watch_heart_rate.csv")
    df = pd.read_csv(path)

    # 컬럼명 통일
    if "time" in df.columns:
        df["timestamp"] = df["time"]
    if "data" in df.columns:
        df["value"] = df["data"]

    # 2) timestamp 처리
    df["timestamp_clean"] = df["timestamp"].str.replace(r"\.\d+Z$", "Z", regex=True)
    df["ts_utc"] = pd.to_datetime(df["timestamp_clean"], utc=True)
    df["ts_kst"] = df["ts_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    # 3) week cutoffs
    cutoffs = sorted([(wk, ts) for wk, ts in last_update_map.items() if wk > 0], key=lambda x: x[0])

    # 4) raw 로그에 week_label 부여
    df["week_label"] = df["ts_kst"].apply(lambda x: assign_week_label(x, cutoffs))

    # ─ 디버깅: 각 주차 구간(raw) 별 데이터 개수 출력 ─────────────────────────
    max_week = cutoffs[-1][0]
    print("===== Raw 로그 주차별 개수 =====")
    for wk in range(max_week):
        cnt = df[df["week_label"] == wk].shape[0]
        print(f"{wk}-{wk+1}주차: {cnt}건")
    print("===============================\n")

    # 5) 1분 단위 평균
    df_min = df.set_index("ts_kst")["value"].resample("min").mean().to_frame("HR_mean")

    # 6) df_min에 week_label 부여
    df_min = df_min.reset_index().rename(columns={"ts_kst": "ts_kst_min"})
    df_min["week_label"] = df_min["ts_kst_min"].apply(lambda x: assign_week_label(x, cutoffs))
    df_min = df_min.set_index("ts_kst_min")

    # ─ 디버깅: 각 주차 구간(1분 데이터) 별 데이터 개수 출력 ─────────────────
    print("===== 실제 측정된 분 개수 (HR_mean not NaN) =====")
    for wk in range(max_week):
        cnt_valid = df_min[df_min["week_label"] == wk]["HR_mean"].count()
        print(f"{wk}-{wk+1}주차: {cnt_valid}건")
    print("==============================================")

    # 7) 파라미터 설정
    resting_hr  = 60   # 안정 심박수 기준 (60)
    night_hours = set(range(0, 6)) | set(range(22, 24))  # 0~5시, 22~23시
    day_hours   = set(range(6, 22))                     # 6~21시

    feature_rows = []

    # 8) 주차별 피처 계산 반복
    for wk in sorted(df_min["week_label"].dropna().unique()):
        df_week = df_min[df_min["week_label"] == wk].copy()
        if df_week.empty:
            continue  # 해당 주차에 데이터가 없으면 건너뜀

        hr_series = df_week["HR_mean"]

        # --- 기초 통계량 계산 ---
        hr_mean = hr_series.mean()      # HR_Mean: 주차 전체 평균 분당 HR
        hr_std  = hr_series.std()       # HR_STD: 주차 전체 분당 HR 표준편차
        hr_max  = hr_series.max()       # HR_Max: 주차 전체 분당 HR 최대값

        # --- HR 변화량(diff) 기반 피처 (전체/블록별 비율 + 평균) ---
        diffs = hr_series.diff().abs().dropna()                       # 연속 분차 분당 평균 HR 절댓값
        weekly_abs_diff_mean = diffs.mean() if not diffs.empty else 0
        hr_diff_mean = weekly_abs_diff_mean                            # HR_DiffMean: 절댓값 변화량 평균

        HR_DiffRatio = diffs[diffs > weekly_abs_diff_mean].shape[0] / len(diffs) \
            if len(diffs) > 0 else np.nan

        # --- 블록별(HR_DiffRatio + Missing) ---
        diffs_hours = diffs.index.hour

        # 아침(06~11시)
        diffs_morning = diffs[diffs_hours.isin(range(6, 12))].dropna()
        if len(diffs_morning) > 0:
            HR_DiffRatio_Morning = diffs_morning[diffs_morning > weekly_abs_diff_mean].shape[0] / len(diffs_morning)
            HR_DiffRatio_Morning_Missing = 0
        else:
            HR_DiffRatio_Morning = 0
            HR_DiffRatio_Morning_Missing = 1

        # 오후(12~17시)
        diffs_afternoon = diffs[diffs_hours.isin(range(12, 18))].dropna()
        if len(diffs_afternoon) > 0:
            HR_DiffRatio_Afternoon = diffs_afternoon[diffs_afternoon > weekly_abs_diff_mean].shape[0] / len(diffs_afternoon)
            HR_DiffRatio_Afternoon_Missing = 0
        else:
            HR_DiffRatio_Afternoon = 0
            HR_DiffRatio_Afternoon_Missing = 1

        # 저녁(18~23시)
        diffs_evening = diffs[diffs_hours.isin(range(18, 24))].dropna()
        if len(diffs_evening) > 0:
            HR_DiffRatio_Evening = diffs_evening[diffs_evening > weekly_abs_diff_mean].shape[0] / len(diffs_evening)
            HR_DiffRatio_Evening_Missing = 0
        else:
            HR_DiffRatio_Evening = 0
            HR_DiffRatio_Evening_Missing = 1

        # 밤(00~05시)
        diffs_night = diffs[diffs_hours.isin(range(0, 6))].dropna()
        if len(diffs_night) > 0:
            HR_DiffRatio_Night = diffs_night[diffs_night > weekly_abs_diff_mean].shape[0] / len(diffs_night)
            HR_DiffRatio_Night_Missing = 0
        else:
            HR_DiffRatio_Night = 0
            HR_DiffRatio_Night_Missing = 1

        # --- 야간 vs 주간 심박 차이 (블록별 평균 + Missing) ---
        df_week["hour"] = df_week.index.hour

        # 주간(06~21시)
        hr_day_vals = df_week[df_week["hour"].isin(day_hours)]["HR_mean"].dropna()
        if len(hr_day_vals) > 0:
            hr_day_mean = hr_day_vals.mean()
            HR_Day_Missing = 0
        else:
            hr_day_mean = 0
            HR_Day_Missing = 1

        # 밤(00~05시, 22~23시)
        hr_night_vals = df_week[df_week["hour"].isin(night_hours)]["HR_mean"].dropna()
        if len(hr_night_vals) > 0:
            hr_night = hr_night_vals.mean()
            HR_Night_Missing = 0
        else:
            hr_night = 0
            HR_Night_Missing = 1

        HR_NightDayGap = hr_night - hr_day_mean  # HR_NightDayGap

        # --- 안정성 지표 ---
        hr_stability = hr_mean / hr_std if hr_std and not np.isnan(hr_std) else np.nan  # HR_Stability

        # --- 평균 대비 안정 심박수 비율---
        hr_resting_hr_ratio = hr_mean / resting_hr if resting_hr else np.nan  # HR_RestingHRRatio

        # --- 블록별(아침/오후/저녁/밤) 평균 HR + Missing 플래그 ---
        # 아침(06~11시)
        block_morning_vals = hr_series[df_week["hour"].isin(range(6, 12))].dropna()
        if len(block_morning_vals) > 0:
            hr_morning = block_morning_vals.mean()
            HR_Morning_Missing = 0
        else:
            hr_morning = 0
            HR_Morning_Missing = 1

        # 오후(12~17시)
        block_afternoon_vals = hr_series[df_week["hour"].isin(range(12, 18))].dropna()
        if len(block_afternoon_vals) > 0:
            hr_afternoon = block_afternoon_vals.mean()
            HR_Afternoon_Missing = 0
        else:
            hr_afternoon = 0
            HR_Afternoon_Missing = 1

        # 저녁(18~23시)
        block_evening_vals = hr_series[df_week["hour"].isin(range(18, 24))].dropna()
        if len(block_evening_vals) > 0:
            hr_evening = block_evening_vals.mean()
            HR_Evening_Missing = 0
        else:
            hr_evening = 0
            HR_Evening_Missing = 1

        # 밤(00~05시)
        block_night_vals = hr_series[df_week["hour"].isin(range(0, 6))].dropna()
        if len(block_night_vals) > 0:
            hr_night = block_night_vals.mean()
            HR_Night_Missing = 0
        else:
            hr_night = 0
            HR_Night_Missing = 1

        # --- 시간대별(0~23시) 평균 HR 벡터 생성 및 결측 플래그 ---
        hourly_mean_raw     = df_week.groupby(df_week.index.hour)["HR_mean"].mean().reindex(range(24), fill_value=np.nan)
        hourly_missing_flag = hourly_mean_raw.isna().astype(int)  # HR_Hour_n_Missing
        hourly_mean_filled  = hourly_mean_raw.fillna(0)           # HR_Hour_n

        # --- 요일별(월=0~일=6) 평균 HR 벡터 생성 및 결측 플래그 ---
        df_week["weekday"]   = df_week.index.weekday  # 실제 날짜 기준 요일 (0=월~6=일)
        weekday_mean_raw     = df_week.groupby("weekday")["HR_mean"].mean().reindex(range(7), fill_value=np.nan)
        weekday_missing_flag = weekday_mean_raw.isna().astype(int)  # HR_Weekday_n_Missing
        weekday_mean_filled  = weekday_mean_raw.fillna(0)           # HR_Weekday_n

        # --- 주말 심박 비율 (토/일 평균 HR 합 / 전체 주차 평균 HR) ---
        weekend_vals     = pd.Series({
            5: weekday_mean_filled.iloc[5],  # Saturday
            6: weekday_mean_filled.iloc[6]   # Sunday
        })
        weekend_mean     = weekend_vals.mean()
        hr_weekend_ratio = weekend_mean / hr_mean if hr_mean and not np.isnan(hr_mean) else np.nan  # HR_WeekendRatio

        # --- 시간대별 평균 HR 분포 엔트로피 ---
        hourly_probs = hourly_mean_filled.copy()
        total_hourly = hourly_probs.sum()
        if total_hourly > 0:
            hourly_probs   = hourly_probs / total_hourly
            hr_hour_entropy = -(hourly_probs[hourly_probs > 0] * np.log(hourly_probs[hourly_probs > 0])).sum()
        else:
            hr_hour_entropy = 0  # HR_HourEntropy

        # --- 피크 시간대 및 피크 심박수 ---
        if not hourly_mean_filled.isna().all():
            hr_peak_hour = int(hourly_mean_filled.idxmax())   # HR_Peak_Hour
            hr_peak_val  = float(hourly_mean_filled.max())    # HR_Peak_Value
        else:
            hr_peak_hour = np.nan
            hr_peak_val  = np.nan

        # --- HR_HighRatio (주차 전체 평균 HR보다 높은 비율) ---
        weekly_mean_hr  = hr_mean
        mask_high_total = hr_series > weekly_mean_hr
        HR_HighRatio    = mask_high_total.sum() / len(hr_series) if len(hr_series) > 0 else np.nan

        hrs_values = hr_series.values
        hrs_hours  = df_week["hour"].values

        # 아침(06~11시)
        vals_morning = hrs_values[(hrs_hours >= 6) & (hrs_hours < 12)]
        if len(vals_morning) > 0:
            HR_HighRatio_Morning = (vals_morning > weekly_mean_hr).sum() / len(vals_morning)
            HR_HighRatio_Morning_Missing = 0
        else:
            HR_HighRatio_Morning = 0
            HR_HighRatio_Morning_Missing = 1

        # 오후(12~17시)
        vals_afternoon = hrs_values[(hrs_hours >= 12) & (hrs_hours < 18)]
        if len(vals_afternoon) > 0:
            HR_HighRatio_Afternoon = (vals_afternoon > weekly_mean_hr).sum() / len(vals_afternoon)
            HR_HighRatio_Afternoon_Missing = 0
        else:
            HR_HighRatio_Afternoon = 0
            HR_HighRatio_Afternoon_Missing = 1

        # 저녁(18~23시)
        vals_evening = hrs_values[(hrs_hours >= 18) & (hrs_hours < 24)]
        if len(vals_evening) > 0:
            HR_HighRatio_Evening = (vals_evening > weekly_mean_hr).sum() / len(vals_evening)
            HR_HighRatio_Evening_Missing = 0
        else:
            HR_HighRatio_Evening = 0
            HR_HighRatio_Evening_Missing = 1

        # 밤(00~05시)
        vals_night = hrs_values[(hrs_hours >= 0) & (hrs_hours < 6)]
        if len(vals_night) > 0:
            HR_HighRatio_Night = (vals_night > weekly_mean_hr).sum() / len(vals_night)
            HR_HighRatio_Night_Missing = 0
        else:
            HR_HighRatio_Night = 0
            HR_HighRatio_Night_Missing = 1

        # --- 피처 딕셔너리 생성 ---
        row = {
            "uid": UID,
            "week": wk,
            "HR_Mean": hr_mean,                            # 주차 전체 평균 분당 HR
            "HR_STD": hr_std,                              # 주차 전체 분당 HR 표준편차
            "HR_Max": hr_max,                              # 주차 전체 분당 HR 최대값
            "HR_DiffMean": hr_diff_mean,                   # 분당 HR 변화량 절댓값 평균
            # HR_NightDayGap 및 Missing 플래그
            "HR_NightDayGap": hr_night - hr_day_mean,      # 야간 평균 - 주간 평균
            "HR_Night_Missing": HR_Night_Missing,          # 밤(00~05시,22~23시)에 데이터 전무 시 1, 아니면 0
            "HR_Day_Missing": HR_Day_Missing,              # 주간(06~21시)에 데이터 전무 시 1, 아니면 0
            "HR_Stability": hr_stability,                  # 평균 / 표준편차 (안정성)
            "HR_RestingHRRatio": hr_resting_hr_ratio,      # 평균 HR / resting_hr(60) 비율
            # 시간대별 평균 HR 및 Missing Flag
            "HR_Morning": hr_morning,                      # 06~11시 평균 분당 HR (없으면 0)
            "HR_Morning_Missing": HR_Morning_Missing,      # 해당 시간대 데이터 전무 시 1
            "HR_Afternoon": hr_afternoon,                  # 12~17시 평균 분당 HR (없으면 0)
            "HR_Afternoon_Missing": HR_Afternoon_Missing,  # 해당 시간대 데이터 전무 시 1
            "HR_Evening": hr_evening,                      # 18~23시 평균 분당 HR (없으면 0)
            "HR_Evening_Missing": HR_Evening_Missing,      # 해당 시간대 데이터 전무 시 1
            "HR_Night": hr_night,                          # 00~05시 평균 분당 HR (없으면 0)
            "HR_Night_Missing_Time": HR_Night_Missing,     # 해당 시간대 데이터 전무 시 1
            "HR_WeekendRatio": hr_weekend_ratio,           # 토/일 평균 HR 합 / 전체 평균 HR
            "HR_HourEntropy": hr_hour_entropy,             # 시간대별 평균 HR 분포 엔트로피
            "HR_Peak_Hour": hr_peak_hour,                  # 피크 평균 HR이 발생한 시간대
            "HR_Peak_Value": hr_peak_val                   # 피크 평균 HR 값
        }

        # --- HR_DiffRatio 전체/시간대별 및 Missing Flag ---
        row["HR_DiffRatio"] = HR_DiffRatio
        row["HR_DiffRatio_Morning"] = HR_DiffRatio_Morning
        row["HR_DiffRatio_Morning_Missing"] = HR_DiffRatio_Morning_Missing
        row["HR_DiffRatio_Afternoon"] = HR_DiffRatio_Afternoon
        row["HR_DiffRatio_Afternoon_Missing"] = HR_DiffRatio_Afternoon_Missing
        row["HR_DiffRatio_Evening"] = HR_DiffRatio_Evening
        row["HR_DiffRatio_Evening_Missing"] = HR_DiffRatio_Evening_Missing
        row["HR_DiffRatio_Night"] = HR_DiffRatio_Night
        row["HR_DiffRatio_Night_Missing"] = HR_DiffRatio_Night_Missing

        # --- HR_HighRatio 전체/시간대별 및 Missing Flag ---
        row["HR_HighRatio"] = HR_HighRatio
        row["HR_HighRatio_Morning"] = HR_HighRatio_Morning
        row["HR_HighRatio_Morning_Missing"] = HR_HighRatio_Morning_Missing
        row["HR_HighRatio_Afternoon"] = HR_HighRatio_Afternoon
        row["HR_HighRatio_Afternoon_Missing"] = HR_HighRatio_Afternoon_Missing
        row["HR_HighRatio_Evening"] = HR_HighRatio_Evening
        row["HR_HighRatio_Evening_Missing"] = HR_HighRatio_Evening_Missing
        row["HR_HighRatio_Night"] = HR_HighRatio_Night
        row["HR_HighRatio_Night_Missing"] = HR_HighRatio_Night_Missing

        # --- 시간대별 평균 HR 벡터 및 Missing Flag 추가 (HR_Hour_0 ~ HR_Hour_23) ---
        for hour in range(24):
            row[f"HR_Hour_{hour}"]         = float(hourly_mean_filled.iloc[hour])      # 해당 시간대 평균 HR (NaN→0)
            row[f"HR_Hour_{hour}_Missing"] = int(hourly_missing_flag.iloc[hour])       # 원본 데이터 전무 시 1, 아니면 0

        # --- 요일별 평균 HR 벡터 및 Missing Flag 추가 (HR_Weekday_0 ~ HR_Weekday_6) ---
        for wd in range(7):
            row[f"HR_Weekday_{wd}"]         = float(weekday_mean_filled.iloc[wd])      # 해당 요일 평균 HR (NaN→0)
            row[f"HR_Weekday_{wd}_Missing"] = int(weekday_missing_flag.iloc[wd])       # 원본 데이터 전무 시 1, 아니면 0

        feature_rows.append(row)

    # 9) 최종 DataFrame 생성 및 디버깅 출력
    final_hr_df = pd.DataFrame(feature_rows)
    print("===== 주차별 심박수 피처 (디버깅용) =====")
    print(final_hr_df)
    print("=====================================\n")

    # 10) CSV로 저장
    output_dir = os.path.join(BASE_DIR, UID, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "watch_heart_rate_processed.csv")

    final_hr_df.to_csv(output_path, index=False)
    print(f"Processed watch_heart_rate feature 파일 저장 완료: {output_path}")
