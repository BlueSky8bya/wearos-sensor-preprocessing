"""PPG SQI(품질지표) 계산 (전처리 단계의 QC)
원본: ppg_data_visualization.ipynb
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

def calculate_rr_noise(rr_intervals, config=None):
    """
    (1) RR 기반 4가지 노이즈 기법 (3σ, 히스토그램, Poincaré, 변동 범위)을 사용해
        'RR 구간의 노이즈 비율(rr_noise_ratio)'을 계산하고, 이를 가중 평균하여 반환.

    반환:
      rr_noise_ratio (float): 0~1 범위 (RR 기반 노이즈 정도)
      detail_dict_rr (dict) : 각 기법별 노이즈 정보
    """
    if config is None:
        config = {}

    sigma_factor        = config.get('sigma_factor', 3.0)         # 3σ 배수
    hist_threshold      = config.get('hist_threshold', 0.15)      # 최빈값 대비 오차 비율
    ellipse_confidence  = config.get('ellipse_confidence', 0.90)  # Poincaré 신뢰구간(기본 90%)
    var_range_threshold = config.get('var_range_threshold', 2.0)  # 변동 범위 임계값
    weights             = config.get('weights', [2.0, 3.0, 1.0, 0.5])
    # ↑ [w_sigma, w_hist, w_ellipse, w_varrange]

    # RR이 너무 적으면 => 노이즈 100% 처리
    if len(rr_intervals) < 2:
        detail_dict_rr = {
            'sigma_noise_ratio': 1.0,
            'hist_noise_ratio': 1.0,
            'ellipse_noise_ratio': 1.0,
            'varrange_noise_ratio': 1.0
        }
        return 1.0, detail_dict_rr

    # RR 통계값
    mean_rr = np.mean(rr_intervals)
    std_rr  = np.std(rr_intervals)

    # ---------------------------
    # (1) 3σ 기반
    # ---------------------------
    upper_bound = mean_rr + sigma_factor * std_rr
    lower_bound = mean_rr - sigma_factor * std_rr
    sigma_outliers = np.sum((rr_intervals < lower_bound) | (rr_intervals > upper_bound))
    sigma_noise_ratio = sigma_outliers / len(rr_intervals)

    # ---------------------------
    # (2) 히스토그램(최빈값) 기반
    # ---------------------------
    bins = np.arange(np.min(rr_intervals), np.max(rr_intervals) + 0.05, 0.05)
    hist, bin_edges = np.histogram(rr_intervals, bins=bins)
    max_bin_index = np.argmax(hist)
    mode_value = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1]) / 2.0
    deviations = np.abs(rr_intervals - mode_value)
    noise_count_hist = np.sum(deviations > hist_threshold * mode_value)
    hist_noise_ratio = noise_count_hist / len(rr_intervals)

    # ---------------------------
    # (3) Poincaré(타원형 분포) 기반
    # ---------------------------
    ellipse_threshold = chi2.ppf(ellipse_confidence, df=2)
    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    points = np.column_stack((x, y))
    mean_point = np.mean(points, axis=0)
    cov_matrix = np.cov(points, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        diff_ = points - mean_point
        m_distances = np.sqrt(np.sum(diff_ @ inv_cov * diff_, axis=1))
        noise_count_ellipse = np.sum(m_distances > ellipse_threshold)
        ellipse_noise_ratio = noise_count_ellipse / len(m_distances)
    except np.linalg.LinAlgError:
        ellipse_noise_ratio = 0.0

    # ---------------------------
    # (4) 변동 범위(MxDMn) 기반
    # ---------------------------
    rr_min = np.min(rr_intervals)
    rr_max = np.max(rr_intervals)
    var_range = rr_max - rr_min
    if var_range > var_range_threshold:
        varrange_noise_ratio = min(1.0, (var_range - var_range_threshold) / var_range)
    else:
        varrange_noise_ratio = 0.0

    # ---------------------------
    # (5) 4가지 기법 가중 평균
    # ---------------------------
    weighted_sum = (weights[0]*sigma_noise_ratio +
                    weights[1]*hist_noise_ratio +
                    weights[2]*ellipse_noise_ratio +
                    weights[3]*varrange_noise_ratio)
    weight_total = sum(weights)
    rr_noise_ratio = weighted_sum / weight_total

    detail_dict_rr = {
        'sigma_noise_ratio': sigma_noise_ratio,
        'hist_noise_ratio': hist_noise_ratio,
        'ellipse_noise_ratio': ellipse_noise_ratio,
        'varrange_noise_ratio': varrange_noise_ratio
    }
    return rr_noise_ratio, detail_dict_rr

def calculate_ppg_var_diff_noise(ppg_signal, config=None):
    """
    (3) PPG 동적(적응형) 임계값 + 미분 기반 노이즈 계산 (윈도우 없이 전체 샘플 단위)

    - 동적 임계값:
      구간 전체 PPG 신호의 전역 통계를 구한 뒤, factor 배만큼을 임계값으로 사용.
      이 임계값을 초과하는 샘플들의 비율을 0~1로 산출.

    - 미분 기반:
      전 구간 미분 신호의 표준편차(또는 평균 등)에 factor를 곱해 임계값 설정.
      초과 샘플 비율을 0~1로 산출.

    반환:
      ppg_noise_ratio (float): 0~1 (분산+미분 종합 노이즈 정도)
      detail_dict_ppg (dict) : 세부 정보
    """
    if config is None:
        config = {}

    # -------------------------------
    # (A) 파라미터 로드
    # -------------------------------
    use_variance_noise   = config.get('use_variance_noise', True)
    dynamic_factor       = config.get('dynamic_factor', 1.5)   # 전역 통계 대비 배수
    use_derivative_noise = config.get('use_derivative_noise', True)
    dynamic_diff_factor  = config.get('dynamic_diff_factor', 1.5)

    # 최종 가중치 (분산 vs 미분)
    var_diff_weights = config.get('var_diff_weights', [1.0, 5.0])
    var_weight, diff_weight = var_diff_weights[0], var_diff_weights[1]

    # -------------------------------
    # (B) 전역 통계 계산
    # -------------------------------
    if ppg_signal is not None and len(ppg_signal) > 0:
        global_mean = np.mean(ppg_signal)
        global_std  = np.std(ppg_signal)
        global_max  = np.max(ppg_signal)
        n_samples   = len(ppg_signal)
    else:
        global_mean = 0.0
        global_std  = 0.0
        global_max  = 0.0
        n_samples   = 0

    # 미분 신호 전역 통계
    diff_signal = None
    if ppg_signal is not None and len(ppg_signal) > 1:
        diff_signal = np.diff(ppg_signal)
        global_diff_std = np.std(diff_signal)
        n_diff_samples  = len(diff_signal)
    else:
        global_diff_std = 0.0
        n_diff_samples  = 0

    # -------------------------------
    # (C) 분산(동적) 기반 (전체 샘플 단위)
    # -------------------------------
    variance_noise_ratio = 0.0
    if use_variance_noise and n_samples > 0:
        # 전역 평균, 표준편차 계산
        lower_bound = global_mean - dynamic_factor * global_std
        upper_bound = global_mean + dynamic_factor * global_std

        # 범위 밖 샘플 개수
        exceed_count = np.sum((ppg_signal < lower_bound) | (ppg_signal > upper_bound))

        # 전체 샘플 대비 비율
        variance_noise_ratio = exceed_count / n_samples

    # -------------------------------
    # (D) 미분(동적) 기반 (전체 샘플 단위)
    # -------------------------------
    derivative_noise_ratio = 0.0
    if use_derivative_noise and diff_signal is not None and n_diff_samples > 0:
        # 전역 미분 신호의 평균과 표준편차
        global_diff_mean = np.mean(diff_signal)
        # 표준편차 * 배수 => 임계값
        dynamic_diff_threshold = global_diff_std * dynamic_diff_factor

        # "평균 ± 임계값" 범위를 벗어나는 샘플 수
        exceed_diff_count = np.sum(
            np.abs(diff_signal - global_diff_mean) > dynamic_diff_threshold
        )

        # 전체 미분 샘플 대비 비율
        derivative_noise_ratio = exceed_diff_count / n_diff_samples

    # -------------------------------
    # (E) 최종 ppg_noise_ratio (가중 평균)
    # -------------------------------
    total_weight = var_weight + diff_weight
    if total_weight > 0:
        ppg_noise_ratio = (
            var_weight * variance_noise_ratio +
            diff_weight * derivative_noise_ratio
        ) / total_weight
    else:
        ppg_noise_ratio = 0.0

    # -------------------------------
    # (F) detail_dict_ppg
    # -------------------------------
    detail_dict_ppg = {
        'variance_noise_ratio': variance_noise_ratio,
        'derivative_noise_ratio': derivative_noise_ratio,
        'ppg_noise_ratio': ppg_noise_ratio,

        # # 전역 통계 로그용
        # 'global_mean': global_mean,
        # 'global_std':  global_std,
        # 'global_max':  global_max,
        # 'global_diff_std': global_diff_std
    }

    return ppg_noise_ratio, detail_dict_ppg

def calculate_amp_peak_noise(ppg_signal, peak_count, config=None):
    """
    (2) 진폭(amplitude) 기반 + 피크(peak) 개수 기반 노이즈 판별
        - (5) 진폭 기반:
            ppg_signal 표준편차가 너무 작거나(amp_threshold_low 미만)
            너무 크면(amp_threshold_high 초과) => 노이즈=1
        - (6) 피크 개수 기반:
            peak_count < min_peak_count or > max_peak_count => 노이즈=1

    반환:
      amplitude_noise_flag (int): 0 or 1
      peakcount_noise_flag (int): 0 or 1
      detail_dict_amppeak (dict)
    """
    if config is None:
        config = {}

    amp_threshold_low  = config.get('amplitude_threshold_low', 100)
    amp_threshold_high = config.get('amplitude_threshold_high', 200000)
    min_peak_count     = config.get('min_peak_count', 5)
    max_peak_count     = config.get('max_peak_count', 200)

    amplitude_noise_flag = 0
    peakcount_noise_flag = 0

    # (5) 진폭 기반 => 표준편차가 범위 밖이면 노이즈=1
    if ppg_signal is not None and len(ppg_signal) > 0:
        amp_std = np.std(ppg_signal)
        if amp_std < amp_threshold_low or amp_std > amp_threshold_high:
            amplitude_noise_flag = 1

    # (6) 피크 개수 기반 => 범위 밖이면 노이즈=1
    if peak_count is not None:
        if (peak_count < min_peak_count) or (peak_count > max_peak_count):
            peakcount_noise_flag = 1

    detail_dict_amppeak = {
        'amplitude_noise_flag': amplitude_noise_flag,
        'peakcount_noise_flag': peakcount_noise_flag
    }
    return amplitude_noise_flag, peakcount_noise_flag, detail_dict_amppeak

def calculate_noise(rr_intervals, ppg_signal=None, peak_count=None, config=None):
    """
    (4) 최종 노이즈 계산 함수
        1) RR 기반 노이즈(rr_noise_ratio)
        2) 진폭/피크 노이즈(둘 중 하나라도 1이면 -> 즉시 노이즈)
        3) PPG 분산/미분 노이즈(ppg_noise_ratio)

        - RR vs PPG도 가중 평균하여 combined_noise_ratio
        - amplitude_noise_flag or peakcount_noise_flag가 1이면 max()로 최종 1
    """
    if config is None:
        config = {}

    # -----------------------
    # (A) RR 기반 노이즈
    # -----------------------
    rr_noise_ratio, detail_rr = calculate_rr_noise(rr_intervals, config=config)

    # -----------------------
    # (B) 진폭/피크 노이즈 플래그
    # -----------------------
    amplitude_noise_flag, peakcount_noise_flag, detail_amppeak = \
        calculate_amp_peak_noise(ppg_signal, peak_count, config=config)

    # -----------------------
    # (C) PPG 분산/미분 기반
    # -----------------------
    ppg_noise_ratio, detail_ppg = calculate_ppg_var_diff_noise(ppg_signal, config=config)
    variance_noise_ratio   = detail_ppg['variance_noise_ratio']
    derivative_noise_ratio = detail_ppg['derivative_noise_ratio']

    # -----------------------
    # (D) RR vs PPG 가중치
    # -----------------------
    rr_weight  = config.get('rr_weight', 1.0)
    ppg_weight = config.get('ppg_weight', 1.0)

    if (rr_weight + ppg_weight) > 0:
        combined_noise_ratio = (
            rr_weight * rr_noise_ratio +
            ppg_weight * ppg_noise_ratio
        ) / (rr_weight + ppg_weight)
    else:
        combined_noise_ratio = rr_noise_ratio

    # -----------------------
    # (F) 최종 노이즈 결정
    # -----------------------
    #   - amplitude_noise_flag=1 or peakcount_noise_flag=1 이면 => max=1
    final_noise_ratio = max(
        combined_noise_ratio,
        amplitude_noise_flag,
        peakcount_noise_flag
    )

    # -----------------------
    # (G) detail_dict 합치기
    # -----------------------
    detail_dict = {}
    detail_dict.update(detail_rr)       # RR 기반
    detail_dict.update(detail_amppeak)  # 진폭/피크
    detail_dict.update(detail_ppg)      # PPG 분산/미분

    # 주요 값 기록
    detail_dict['rr_noise_ratio']       = rr_noise_ratio
    detail_dict['variance_noise_ratio'] = variance_noise_ratio
    detail_dict['derivative_noise_ratio'] = derivative_noise_ratio
    detail_dict['ppg_noise_ratio']      = ppg_noise_ratio
    detail_dict['combined_noise_ratio'] = combined_noise_ratio
    detail_dict['amplitude_noise_flag'] = amplitude_noise_flag
    detail_dict['peakcount_noise_flag'] = peakcount_noise_flag
    detail_dict['final_noise_ratio']    = final_noise_ratio

    return final_noise_ratio, detail_dict

def calculate_SQI(rr_intervals, ppg_signal=None, peak_count=None, config=None):
    """
    (5) SQI = 1 - final_noise_ratio
    """
    final_noise_ratio, detail_dict = calculate_noise(
        rr_intervals=rr_intervals,
        ppg_signal=ppg_signal,
        peak_count=peak_count,
        config=config
    )
    sqi = 1 - final_noise_ratio
    return sqi, final_noise_ratio, detail_dict
