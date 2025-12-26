"""걸음수(Steps) 전처리 (SOP 전처리 범위)
원본: 전처리(response, heart rate, step count).ipynb

포함 범위:
- 분 단위 변환/세션 구성
- 누적값→증분 계산 및 기본 정합성 보정
- 이상치 탐지(전처리 QC)

제외:
- 일/주 단위 요약 feature 계산
- 그룹 비교/통계/시각화 리포트
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def preprocess_watch_step_count():
    """
    watch_step_count.csv 파일을 전처리하여, 주차별 다양한 걸음수 특성(feature)을 추출하고
    'watch_step_count_processed.csv'로 저장합니다.

    주요 단계:
      1) 정제된 세션 데이터 읽기 (이미 전처리된 파일 사용)
      2) timestamp UTC → KST(naive) 변환
      3) 주차별 경계 시점(cutoffs) 설정 및 week_label 부여
      4) 세션별 1분 단위 걸음수 증분 계산
      5) 1분 단위 데이터에 week_label 부여
      6) 주차별로 다음 피처 계산:
         - 기본 통계량 (평균, 표준편차, 최대값, 총합 등)
         - 시간대별 활동량 (아침/오후/저녁/밤)
         - 요일별 활동 패턴
         - 활동 집중도 및 분포 특성
         - 세션 기반 특성 (지속시간, 강도 등)
      7) (uid, week)별로 피처를 하나의 DataFrame에 모아서 저장
    """

    # 1) 정제된 세션 데이터 읽기
    sessions_path = os.path.join(BASE_DIR, UID, "processed", "watch_step_count_sessions_normalized_minutes.csv")
    filtered_path = os.path.join(BASE_DIR, UID, "processed", "watch_step_count_continuous_sessions.csv")

    if not os.path.exists(sessions_path) or not os.path.exists(filtered_path):
        print("정제된 걸음수 데이터 파일이 없습니다. 먼저 전처리를 실행해주세요.")
        return

    df_sessions = pd.read_csv(sessions_path)
    df_filtered = pd.read_csv(filtered_path)

    # timestamp 컬럼을 datetime으로 변환
    df_sessions["ts_minute"] = pd.to_datetime(df_sessions["ts_minute"])
    df_filtered["ts_kst"] = pd.to_datetime(df_filtered["ts_kst"])

    # NaN 값 처리
    df_sessions["cum_steps_per_min"] = df_sessions["cum_steps_per_min"].fillna(0)
    df_filtered["normalized_value"] = df_filtered["normalized_value"].fillna(0)

    # 2) 주차별 경계 시점(cutoffs) 리스트 생성 (week > 0만 포함)
    cutoffs = sorted(
        [(wk, ts) for wk, ts in last_update_map.items() if wk > 0],
        key=lambda x: x[0]
    )

    # 3) 세션별 1분 단위 데이터에 week_label 부여
    df_sessions["week_label"] = df_sessions["ts_minute"].apply(lambda x: assign_week_label(x, cutoffs))

    # 4) 원본 세션 데이터에도 week_label 부여
    df_filtered["week_label"] = df_filtered["ts_kst"].apply(lambda x: assign_week_label(x, cutoffs))

    # 5) 1분 단위 걸음수 증분 계산 (세션별로)
    step_increment_data = []

    for session_id in df_sessions["session_id"].unique():
        session_data = df_sessions[df_sessions["session_id"] == session_id].sort_values("ts_minute")

        # 전 분 대비 증분 계산
        session_data["step_increment"] = session_data["cum_steps_per_min"].diff().fillna(0)
        # 음수 증분은 0으로 처리 (데이터 오류 방지)
        session_data["step_increment"] = session_data["step_increment"].clip(lower=0)
        # NaN 처리 추가
        session_data["step_increment"] = session_data["step_increment"].fillna(0)

        step_increment_data.append(session_data)

    df_increments = pd.concat(step_increment_data, ignore_index=True)

    # 6) 시간 관련 정보 추가
    df_increments["hour"] = df_increments["ts_minute"].dt.hour
    df_increments["weekday"] = df_increments["ts_minute"].dt.weekday  # 0=월~6=일
    df_increments["date"] = df_increments["ts_minute"].dt.date

    # 7) 파라미터 설정
    morning_hours = set(range(6, 12))    # 06~11시
    afternoon_hours = set(range(12, 18)) # 12~17시
    evening_hours = set(range(18, 24))   # 18~23시
    night_hours = set(range(0, 6))       # 00~05시

    feature_rows = []

    # 8) 주차별 피처 계산
    for wk in sorted(df_increments["week_label"].dropna().unique()):
        df_week = df_increments[df_increments["week_label"] == wk].copy()
        df_week_sessions = df_filtered[df_filtered["week_label"] == wk].copy()

        if df_week.empty:
            continue

        # === 기본 걸음수 통계량 (NaN 처리 강화) ===
        step_increments = df_week["step_increment"].fillna(0)
        cum_steps = df_week["cum_steps_per_min"].fillna(0)

        steps_mean = step_increments.mean() if len(step_increments) > 0 else 0
        steps_std = step_increments.std() if len(step_increments) > 0 else 0
        steps_max = step_increments.max() if len(step_increments) > 0 else 0
        steps_total = step_increments.sum() if len(step_increments) > 0 else 0
        steps_median = step_increments.median() if len(step_increments) > 0 else 0

        # NaN 처리
        steps_mean = 0 if pd.isna(steps_mean) else steps_mean
        steps_std = 0 if pd.isna(steps_std) else steps_std
        steps_max = 0 if pd.isna(steps_max) else steps_max
        steps_total = 0 if pd.isna(steps_total) else steps_total
        steps_median = 0 if pd.isna(steps_median) else steps_median

        # === 활동 강도 관련 피처 ===
        # 활동 분 비율 (걸음수 > 0인 분의 비율)
        active_minutes = (step_increments > 0).sum()
        total_minutes = len(step_increments)
        steps_active_ratio = active_minutes / total_minutes if total_minutes > 0 else 0

        # 고강도 활동 비율 (평균보다 높은 걸음수)
        if steps_mean > 0:
            steps_high_ratio = (step_increments > steps_mean).sum() / total_minutes
        else:
            steps_high_ratio = 0

        # 매우 고강도 활동 비율 (평균 + 1*표준편차 초과)
        if steps_std > 0:
            high_threshold = steps_mean + steps_std
            steps_very_high_ratio = (step_increments > high_threshold).sum() / total_minutes
        else:
            steps_very_high_ratio = 0

        # === 걸음수 변화량 관련 피처 ===
        diffs = step_increments.diff().abs().dropna()
        steps_diff_mean = diffs.mean() if not diffs.empty else 0
        steps_diff_std = diffs.std() if not diffs.empty else 0

        # NaN 처리
        steps_diff_mean = 0 if pd.isna(steps_diff_mean) else steps_diff_mean
        steps_diff_std = 0 if pd.isna(steps_diff_std) else steps_diff_std

        # 급격한 변화 비율
        if steps_diff_mean > 0:
            steps_diff_ratio = (diffs > steps_diff_mean).sum() / len(diffs) if len(diffs) > 0 else 0
        else:
            steps_diff_ratio = 0

        # === 시간대별 활동량 및 Missing 플래그 ===
        df_week_with_hour = df_week.copy()

        # 아침 (06~11시)
        morning_data = df_week_with_hour[df_week_with_hour["hour"].isin(morning_hours)]["step_increment"]
        if len(morning_data) > 0:
            steps_morning = morning_data.sum()
            steps_morning_mean = morning_data.mean()
            steps_morning_missing = 0
            # NaN 처리
            steps_morning = 0 if pd.isna(steps_morning) else steps_morning
            steps_morning_mean = 0 if pd.isna(steps_morning_mean) else steps_morning_mean
        else:
            steps_morning = 0
            steps_morning_mean = 0
            steps_morning_missing = 1

        # 오후 (12~17시)
        afternoon_data = df_week_with_hour[df_week_with_hour["hour"].isin(afternoon_hours)]["step_increment"]
        if len(afternoon_data) > 0:
            steps_afternoon = afternoon_data.sum()
            steps_afternoon_mean = afternoon_data.mean()
            steps_afternoon_missing = 0
            # NaN 처리
            steps_afternoon = 0 if pd.isna(steps_afternoon) else steps_afternoon
            steps_afternoon_mean = 0 if pd.isna(steps_afternoon_mean) else steps_afternoon_mean
        else:
            steps_afternoon = 0
            steps_afternoon_mean = 0
            steps_afternoon_missing = 1

        # 저녁 (18~23시)
        evening_data = df_week_with_hour[df_week_with_hour["hour"].isin(evening_hours)]["step_increment"]
        if len(evening_data) > 0:
            steps_evening = evening_data.sum()
            steps_evening_mean = evening_data.mean()
            steps_evening_missing = 0
            # NaN 처리
            steps_evening = 0 if pd.isna(steps_evening) else steps_evening
            steps_evening_mean = 0 if pd.isna(steps_evening_mean) else steps_evening_mean
        else:
            steps_evening = 0
            steps_evening_mean = 0
            steps_evening_missing = 1

        # 밤 (00~05시)
        night_data = df_week_with_hour[df_week_with_hour["hour"].isin(night_hours)]["step_increment"]
        if len(night_data) > 0:
            steps_night = night_data.sum()
            steps_night_mean = night_data.mean()
            steps_night_missing = 0
            # NaN 처리
            steps_night = 0 if pd.isna(steps_night) else steps_night
            steps_night_mean = 0 if pd.isna(steps_night_mean) else steps_night_mean
        else:
            steps_night = 0
            steps_night_mean = 0
            steps_night_missing = 1

        # === 시간대별 고강도 활동 비율 ===
        def calc_high_ratio_for_timeblock(data, weekly_mean):
            if len(data) > 0 and weekly_mean > 0:
                ratio = (data > weekly_mean).sum() / len(data)
                return 0 if pd.isna(ratio) else ratio
            return 0

        steps_high_ratio_morning = calc_high_ratio_for_timeblock(morning_data, steps_mean)
        steps_high_ratio_afternoon = calc_high_ratio_for_timeblock(afternoon_data, steps_mean)
        steps_high_ratio_evening = calc_high_ratio_for_timeblock(evening_data, steps_mean)
        steps_high_ratio_night = calc_high_ratio_for_timeblock(night_data, steps_mean)

        # === 요일별 활동량 ===
        weekday_stats = df_week_with_hour.groupby("weekday")["step_increment"].agg(['sum', 'mean']).reindex(range(7), fill_value=0)
        weekday_missing = df_week_with_hour.groupby("weekday")["step_increment"].count().reindex(range(7), fill_value=0) == 0

        # NaN 처리
        weekday_stats = weekday_stats.fillna(0)

        # 주말 vs 평일 비율
        weekend_steps = weekday_stats.loc[[5, 6], 'sum'].sum()  # 토, 일
        weekday_steps = weekday_stats.loc[range(5), 'sum'].sum()  # 월~금

        # NaN 처리
        weekend_steps = 0 if pd.isna(weekend_steps) else weekend_steps
        weekday_steps = 0 if pd.isna(weekday_steps) else weekday_steps

        if weekday_steps > 0:
            steps_weekend_ratio = weekend_steps / weekday_steps
        else:
            steps_weekend_ratio = 0 if weekend_steps == 0 else 0  # inf 대신 0으로 처리

        # === 시간대별 활동 분포 엔트로피 ===
        hourly_steps = df_week_with_hour.groupby("hour")["step_increment"].sum().reindex(range(24), fill_value=0)
        hourly_steps = hourly_steps.fillna(0)
        total_hourly_steps = hourly_steps.sum()

        if total_hourly_steps > 0:
            hourly_probs = hourly_steps / total_hourly_steps
            steps_hour_entropy = -(hourly_probs[hourly_probs > 0] * np.log(hourly_probs[hourly_probs > 0])).sum()
            steps_hour_entropy = 0 if pd.isna(steps_hour_entropy) else steps_hour_entropy
        else:
            steps_hour_entropy = 0

        # === 피크 활동 시간 및 값 ===
        if not hourly_steps.isna().all() and hourly_steps.sum() > 0:
            steps_peak_hour = int(hourly_steps.idxmax())
            steps_peak_value = float(hourly_steps.max())
            # NaN 처리
            steps_peak_value = 0 if pd.isna(steps_peak_value) else steps_peak_value
        else:
            steps_peak_hour = 0  # NaN 대신 0으로 처리
            steps_peak_value = 0

        # === 세션 기반 특성 ===
        session_ids_in_week = df_week["session_id"].unique()
        session_count = len(session_ids_in_week)

        # 세션별 지속시간과 걸음수
        session_durations = []
        session_step_totals = []
        session_intensities = []

        for sid in session_ids_in_week:
            session_filtered = df_week_sessions[df_week_sessions["session_id"] == sid]
            if not session_filtered.empty:
                start_time = session_filtered["ts_kst"].min()
                end_time = session_filtered["ts_kst"].max()
                duration_minutes = (end_time - start_time).total_seconds() / 60
                session_steps = session_filtered["normalized_value"].max()

                # NaN 처리
                duration_minutes = 0 if pd.isna(duration_minutes) else duration_minutes
                session_steps = 0 if pd.isna(session_steps) else session_steps

                session_durations.append(duration_minutes)
                session_step_totals.append(session_steps)

                # 세션 강도 (분당 평균 걸음수)
                if duration_minutes > 0:
                    intensity = session_steps / duration_minutes
                    session_intensities.append(0 if pd.isna(intensity) else intensity)
                else:
                    session_intensities.append(0)

        # 세션 통계
        if session_durations:
            session_duration_mean = np.mean(session_durations)
            session_duration_std = np.std(session_durations)
            session_duration_max = np.max(session_durations)
            session_steps_mean = np.mean(session_step_totals)
            session_steps_std = np.std(session_step_totals)
            session_intensity_mean = np.mean(session_intensities)
            session_intensity_std = np.std(session_intensities)

            # NaN 처리
            session_duration_mean = 0 if pd.isna(session_duration_mean) else session_duration_mean
            session_duration_std = 0 if pd.isna(session_duration_std) else session_duration_std
            session_duration_max = 0 if pd.isna(session_duration_max) else session_duration_max
            session_steps_mean = 0 if pd.isna(session_steps_mean) else session_steps_mean
            session_steps_std = 0 if pd.isna(session_steps_std) else session_steps_std
            session_intensity_mean = 0 if pd.isna(session_intensity_mean) else session_intensity_mean
            session_intensity_std = 0 if pd.isna(session_intensity_std) else session_intensity_std
        else:
            session_duration_mean = session_duration_std = session_duration_max = 0
            session_steps_mean = session_steps_std = 0
            session_intensity_mean = session_intensity_std = 0

        # === 일별 활동 일관성 ===
        daily_steps = df_week_with_hour.groupby("date")["step_increment"].sum()
        daily_steps = daily_steps.fillna(0)
        if len(daily_steps) > 1:
            daily_std = daily_steps.std()
            daily_std = 0 if pd.isna(daily_std) else daily_std
            steps_daily_consistency = 1 / (1 + daily_std) if daily_std > 0 else 1
        else:
            steps_daily_consistency = 1

        # === 활동 집중도 (Gini coefficient 기반) ===
        sorted_steps = np.sort(step_increments.values)
        n = len(sorted_steps)
        if n > 0 and sorted_steps.sum() > 0:
            cumsum = np.cumsum(sorted_steps)
            steps_gini = (2 * np.sum((np.arange(1, n + 1) * sorted_steps))) / (n * cumsum[-1]) - (n + 1) / n
            steps_gini = 0 if pd.isna(steps_gini) else steps_gini
        else:
            steps_gini = 0

        # === 피처 딕셔너리 생성 ===
        row = {
            "uid": UID,
            "week": wk,

            # 기본 통계량
            "STEPS_Mean": steps_mean,
            "STEPS_STD": steps_std,
            "STEPS_Max": steps_max,
            "STEPS_Total": steps_total,
            "STEPS_Median": steps_median,

            # 활동 강도 관련
            "STEPS_ActiveRatio": steps_active_ratio,
            "STEPS_HighRatio": steps_high_ratio,
            "STEPS_VeryHighRatio": steps_very_high_ratio,

            # 변화량 관련
            "STEPS_DiffMean": steps_diff_mean,
            "STEPS_DiffStd": steps_diff_std,
            "STEPS_DiffRatio": steps_diff_ratio,

            # 시간대별 총 걸음수
            "STEPS_Morning": steps_morning,
            "STEPS_Morning_Missing": steps_morning_missing,
            "STEPS_Afternoon": steps_afternoon,
            "STEPS_Afternoon_Missing": steps_afternoon_missing,
            "STEPS_Evening": steps_evening,
            "STEPS_Evening_Missing": steps_evening_missing,
            "STEPS_Night": steps_night,
            "STEPS_Night_Missing": steps_night_missing,

            # 시간대별 평균 걸음수
            "STEPS_Morning_Mean": steps_morning_mean,
            "STEPS_Afternoon_Mean": steps_afternoon_mean,
            "STEPS_Evening_Mean": steps_evening_mean,
            "STEPS_Night_Mean": steps_night_mean,

            # 시간대별 고강도 활동 비율
            "STEPS_HighRatio_Morning": steps_high_ratio_morning,
            "STEPS_HighRatio_Afternoon": steps_high_ratio_afternoon,
            "STEPS_HighRatio_Evening": steps_high_ratio_evening,
            "STEPS_HighRatio_Night": steps_high_ratio_night,

            # 요일별 특성
            "STEPS_WeekendRatio": steps_weekend_ratio,
            "STEPS_DailyConsistency": steps_daily_consistency,

            # 분포 특성
            "STEPS_HourEntropy": steps_hour_entropy,
            "STEPS_Peak_Hour": steps_peak_hour,
            "STEPS_Peak_Value": steps_peak_value,
            "STEPS_Gini": steps_gini,

            # 세션 기반 특성
            "STEPS_SessionCount": session_count,
            "STEPS_SessionDuration_Mean": session_duration_mean,
            "STEPS_SessionDuration_STD": session_duration_std,
            "STEPS_SessionDuration_Max": session_duration_max,
            "STEPS_SessionSteps_Mean": session_steps_mean,
            "STEPS_SessionSteps_STD": session_steps_std,
            "STEPS_SessionIntensity_Mean": session_intensity_mean,
            "STEPS_SessionIntensity_STD": session_intensity_std,
        }

        # === 시간대별 걸음수 벡터 및 Missing Flag 추가 ===
        for hour in range(24):
            hour_data = df_week_with_hour[df_week_with_hour["hour"] == hour]["step_increment"]
            if len(hour_data) > 0:
                hour_sum = hour_data.sum()
                row[f"STEPS_Hour_{hour}"] = 0 if pd.isna(hour_sum) else hour_sum
                row[f"STEPS_Hour_{hour}_Missing"] = 0
            else:
                row[f"STEPS_Hour_{hour}"] = 0
                row[f"STEPS_Hour_{hour}_Missing"] = 1

        # === 요일별 걸음수 벡터 및 Missing Flag 추가 ===
        for wd in range(7):
            wd_total = weekday_stats.loc[wd, 'sum'] if wd in weekday_stats.index else 0
            wd_missing = int(weekday_missing.loc[wd]) if wd in weekday_missing.index else 1

            # NaN 처리
            wd_total = 0 if pd.isna(wd_total) else wd_total

            row[f"STEPS_Weekday_{wd}"] = wd_total
            row[f"STEPS_Weekday_{wd}_Missing"] = wd_missing

        feature_rows.append(row)

    # 9) 최종 DataFrame 생성 및 NaN 처리
    final_steps_df = pd.DataFrame(feature_rows)

    # 전체 DataFrame에서 남은 NaN 값들을 0으로 처리
    final_steps_df = final_steps_df.fillna(0)

    print("===== 주차별 걸음수 피처 (디버깅용) =====")
    print(final_steps_df.head())
    print(f"생성된 피처 수: {len(final_steps_df.columns)}")
    print(f"NaN 값 확인: {final_steps_df.isna().sum().sum()}")
    print("=====================================\n")

    # 10) CSV로 저장
    output_dir = os.path.join(BASE_DIR, UID, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "watch_step_count_processed.csv")

    final_steps_df.to_csv(output_path, index=False)
    print(f"Processed watch_step_count feature 파일 저장 완료: {output_path}")

    if final_steps_df is not None:
        print("주요 피처 요약:")
        print(f"- 기본 통계량: Mean, STD, Max, Total, Median")
        print(f"- 활동 강도: ActiveRatio, HighRatio, VeryHighRatio")
        print(f"- 시간대별 활동량: Morning, Afternoon, Evening, Night (총량 + 평균 + Missing Flag)")
        print(f"- 요일별 활동량: Weekday_0~6 + Missing Flag")
        print(f"- 시간대별 활동량: Hour_0~23 + Missing Flag")
        print(f"- 분포 특성: HourEntropy, Peak_Hour/Value, Gini coefficient")
        print(f"- 세션 기반: SessionCount, Duration, Steps, Intensity 통계")
        print(f"- 기타: WeekendRatio, DailyConsistency, DiffMean/STD/Ratio")

def create_minute_data(df_normalized):
    """세션별 1분 단위 누적 걸음수 계산 (시작분부터 끝분까지 0부터 정규화)"""
    if df_normalized.empty:
        return pd.DataFrame()

    # 1) 세션별 시작 시각(floor 분 단위) 구하기
    start_times = (
        df_normalized
        .groupby("session_id")["ts_kst"]
        .min()
        .dt.floor("min")
    )

    # 2) 분 단위 누적 데이터 뽑아내기
    df_min = df_normalized.copy()
    df_min["ts_minute"] = df_min["ts_kst"].dt.floor("min")
    session_minutes = (
        df_min
        .groupby(["session_id", "ts_minute"])["normalized_value"]
        .max()
        .rename("cum_steps_per_min")
        .to_frame()
        .reset_index()
    )

    # 3) asfreq → ffill → fillna 로 1분단위 전 구간 채우기
    session_list = []
    for sid, grp in session_minutes.groupby("session_id"):
        grp = grp.set_index("ts_minute").asfreq("min")

        # 시작 분 보장
        start_min = start_times.loc[sid]
        if start_min not in grp.index:
            grp.loc[start_min] = np.nan

        grp = grp.sort_index()
        grp["cum_steps_per_min"] = grp["cum_steps_per_min"].ffill().fillna(0)

        grp = grp.reset_index().assign(session_id=sid)
        session_list.append(grp)

    # 4) 합치고—여기서 baseline(최소값) 빼기!
    df_out = pd.concat(session_list, ignore_index=True) if session_list else pd.DataFrame()
    if not df_out.empty:
        df_out["cum_steps_per_min"] = (
            df_out
            .groupby("session_id")["cum_steps_per_min"]
            .transform(lambda x: x - x.min())
        )

    return df_out

def create_continuous_sessions(df):
    """연속성을 고려한 개선된 세션 분리 로직"""
    print("=== 연속성 기반 세션 분리 시작 ===")

    if len(df) == 0:
        return pd.DataFrame()

    sessions = []
    current_session = []
    session_id = 0

    for idx, row in df.iterrows():
        if len(current_session) == 0:
            # 첫 번째 행이거나 새로운 세션 시작
            current_session.append({
                'ts_kst': row['ts_kst'],
                'value': row['value'],
                'session_id': session_id
            })
        else:
            prev_time = current_session[-1]['ts_kst']
            prev_value = current_session[-1]['value']

            time_gap = (row['ts_kst'] - prev_time).total_seconds()
            value_diff = row['value'] - prev_value

            # 개선된 세션 분리 조건
            should_split = False

            # 1) 시간 간격이 너무 큰 경우 (30분 → 1시간으로 완화)
            if time_gap >= SESSION_GAP_THRESHOLD:
                should_split = True
                print(f"  시간 간격으로 분리: {time_gap/60:.1f}분")

            # 2) 걸음수가 큰 폭으로 역행하는 경우 (허용 범위 추가)
            elif value_diff < -MAX_BACKWARD_TOLERANCE:
                should_split = True
                print(f"  걸음수 역행으로 분리: {value_diff}")

            # 3) 시간이 역행하는 경우
            elif time_gap < 0:
                print(f"  시간 역행 데이터 건너뛰기: {time_gap}")
                continue

            if should_split:
                # 현재 세션이 충분히 길고 의미있는 활동이면 저장
                if len(current_session) > 1:
                    session_df = pd.DataFrame(current_session)
                    duration = (session_df['ts_kst'].max() - session_df['ts_kst'].min()).total_seconds() / 60
                    step_increase = session_df['value'].max() - session_df['value'].min()

                    if duration >= MIN_SESSION_DURATION and step_increase >= MIN_STEPS_FOR_ACTIVITY:
                        sessions.append(session_df)
                        print(f"  세션 {session_id} 저장: {duration:.1f}분, {step_increase}걸음")
                    else:
                        print(f"  세션 {session_id} 제외: {duration:.1f}분, {step_increase}걸음 (기준 미달)")

                # 새로운 세션 시작
                session_id += 1
                current_session = [{
                    'ts_kst': row['ts_kst'],
                    'value': row['value'],
                    'session_id': session_id
                }]
            else:
                # 같은 세션에 추가 (연속성 유지)
                current_session.append({
                    'ts_kst': row['ts_kst'],
                    'value': row['value'],
                    'session_id': session_id
                })

    # 마지막 세션 처리
    if len(current_session) > 1:
        session_df = pd.DataFrame(current_session)
        duration = (session_df['ts_kst'].max() - session_df['ts_kst'].min()).total_seconds() / 60
        step_increase = session_df['value'].max() - session_df['value'].min()

        if duration >= MIN_SESSION_DURATION and step_increase >= MIN_STEPS_FOR_ACTIVITY:
            sessions.append(session_df)
            print(f"  마지막 세션 {session_id} 저장: {duration:.1f}분, {step_increase}걸음")

    if sessions:
        result_df = pd.concat(sessions, ignore_index=True)
        # 세션 ID 재할당 (0부터 시작)
        session_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(result_df['session_id'].unique()))}
        result_df['session_id'] = result_df['session_id'].map(session_mapping)

        print(f"총 {len(sessions)}개 유효한 연속 세션 생성")
        return result_df
    else:
        print("유효한 연속 세션이 없습니다.")
        return pd.DataFrame()

def merge_zero_step_sessions(session_group):
    """연속된 0걸음 세션들을 하나로 병합"""
    if not session_group:
        return None

    print(f"  {len(session_group)}개 연속 0걸음 세션을 병합")

    # 시간순으로 정렬
    session_group.sort(key=lambda x: x['start_time'])

    # 모든 세션의 데이터를 합치기
    all_data = []
    total_duration = 0

    for session in session_group:
        all_data.append(session['data'])
        total_duration += session['duration']

    # 데이터 병합
    merged_data = pd.concat(all_data, ignore_index=True)
    merged_data = merged_data.sort_values('ts_kst').reset_index(drop=True)

    # 전체를 다시 0부터 시작하도록 정규화
    if len(merged_data) > 0:
        start_value = merged_data.iloc[0]['normalized_value']
        merged_data['normalized_value'] = merged_data['normalized_value'] - start_value
        merged_data['normalized_value'] = merged_data['normalized_value'].clip(lower=0)

    # 병합된 세션이 최소 기준을 만족하는지 확인
    if total_duration >= MIN_SESSION_DURATION:
        return {
            'session_id': session_group[0]['session_id'],  # 첫 번째 세션 ID 사용
            'data': merged_data,
            'final_steps': merged_data['normalized_value'].max() if len(merged_data) > 0 else 0,
            'duration': total_duration,
            'start_time': merged_data['ts_kst'].min() if len(merged_data) > 0 else None,
            'end_time': merged_data['ts_kst'].max() if len(merged_data) > 0 else None
        }
    else:
        print(f"  병합된 세션도 지속시간 부족으로 제외: {total_duration:.1f}분")
        return None

def normalize_continuous_sessions(df):
    """연속성을 고려한 정규화 - 각 세션을 0부터 시작하게 하고 연속된 0걸음 세션들을 병합"""
    print("=== 연속성 기반 정규화 시작 (개선된 버전) ===")

    if len(df) == 0:
        return df

    # 1단계: 각 세션을 0부터 시작하도록 정규화
    normalized_sessions = []
    session_info = []  # 세션 정보 저장용

    for session_id in sorted(df['session_id'].unique()):
        session_data = df[df['session_id'] == session_id].copy()
        session_data = session_data.sort_values('ts_kst').reset_index(drop=True)

        if len(session_data) < 2:
            continue

        # 세션의 시작값을 0으로 만들어 정규화 (무조건 0부터 시작)
        start_value = session_data.iloc[0]['value']
        session_data['normalized_value'] = session_data['value'] - start_value

        # 음수 값 처리 (걸음수는 감소할 수 없음)
        session_data['normalized_value'] = session_data['normalized_value'].clip(lower=0)

        # 세션 내 비정상적인 역행 제거
        if len(session_data) > 2:
            valid_indices = [0]
            prev_max = session_data.iloc[0]['normalized_value']

            for i in range(1, len(session_data)):
                current_value = session_data.iloc[i]['normalized_value']

                if current_value >= prev_max or (prev_max - current_value) <= MAX_BACKWARD_TOLERANCE:
                    valid_indices.append(i)
                    prev_max = max(prev_max, current_value)

            session_data = session_data.iloc[valid_indices].reset_index(drop=True)

        # 세션 정보 저장
        final_steps = session_data['normalized_value'].max()
        duration = (session_data['ts_kst'].max() - session_data['ts_kst'].min()).total_seconds() / 60

        session_info.append({
            'session_id': session_id,
            'data': session_data,
            'final_steps': final_steps,
            'duration': duration,
            'start_time': session_data['ts_kst'].min(),
            'end_time': session_data['ts_kst'].max()
        })

        print(f"세션 {session_id}: 실제 걸음수 {final_steps:.0f}걸음, 지속시간 {duration:.1f}분")

    # 2단계: 연속된 0걸음 세션들을 병합
    print("\n=== 연속된 0걸음 세션 병합 시작 ===")
    merged_sessions = []
    current_merge_group = []

    for i, info in enumerate(session_info):
        if info['final_steps'] <= MIN_STEPS_FOR_ACTIVITY:
            # 0걸음 세션 - 병합 그룹에 추가
            current_merge_group.append(info)
            print(f"세션 {info['session_id']}: 0걸음 세션 - 병합 대상")
        else:
            # 유효한 걸음수가 있는 세션
            # 이전에 모인 0걸음 세션들이 있다면 먼저 처리
            if current_merge_group:
                merged_session = merge_zero_step_sessions(current_merge_group)
                if merged_session is not None:
                    merged_sessions.append(merged_session)
                current_merge_group = []

            # 현재 유효한 세션 추가
            if info['duration'] >= MIN_SESSION_DURATION:
                merged_sessions.append(info)
                print(f"세션 {info['session_id']}: 유효한 활동 세션 유지")
            else:
                print(f"세션 {info['session_id']}: 지속시간 부족으로 제외")

    # 마지막에 남은 0걸음 세션들 처리
    if current_merge_group:
        merged_session = merge_zero_step_sessions(current_merge_group)
        if merged_session is not None:
            merged_sessions.append(merged_session)

    # 3단계: 최종 결과 생성
    if merged_sessions:
        final_sessions = []
        for i, session_info in enumerate(merged_sessions):
            session_data = session_info['data'].copy()
            session_data['session_id'] = i  # 새로운 세션 ID 할당
            final_sessions.append(session_data)

        result_df = pd.concat(final_sessions, ignore_index=True)
        print(f"\n정규화 및 병합 완료: {len(merged_sessions)}개 최종 세션")
        return result_df
    else:
        print("유효한 활동 세션이 없습니다.")
        return pd.DataFrame()

def is_reasonable_step_value(value):
    """걸음수 값이 합리적인 범위인지 확인"""
    return 0 <= value <= MAX_REASONABLE_STEPS

def detect_outliers_iqr(values, multiplier=3.0):
    """IQR 방법으로 이상치 탐지"""
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (values < lower_bound) | (values > upper_bound)

def detect_outliers_zscore(values, threshold=3.0):
    """Z-score 방법으로 이상치 탐지"""
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)
    z_scores = np.abs(stats.zscore(values))
    return z_scores > threshold

def detect_sudden_jumps(values, timestamps, max_steps_per_minute=MAX_STEPS_PER_MINUTE):
    """갑작스러운 걸음수 증가 탐지"""
    if len(values) < 2:
        return np.zeros(len(values), dtype=bool)

    outliers = np.zeros(len(values), dtype=bool)

    for i in range(1, len(values)):
        time_diff = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds() / 60.0
        step_diff = values[i] - values[i-1]

        if time_diff > 0:
            steps_per_minute = step_diff / time_diff
            if steps_per_minute > max_steps_per_minute:
                outliers[i] = True

    return outliers
