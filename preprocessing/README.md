# preprocessing (minimal)

웨어러블/스마트폰 기반 데이터의 전처리(정규화·정합성·윈도우링·QC·라벨링)에 필요한 코드 모음입니다.  
(분석용 대량 feature 생성, 통계/리포팅, 시각화 등은 목적 범위 밖으로 두었습니다.)

---

## 1) 포함 범위

### PPG (Green PPG Raw)
- CSV 로딩 및 컬럼 표준화(`time`, `value`)
- 밴드패스 필터(Butterworth band-pass)
- SQI 계산(1분 윈도우 단위 QC에 사용하는 품질지표 계산 로직)

### Heart Rate
- 타임스탬프 정규화(UTC epoch(ms) → KST)
- 1분 단위 집계 전처리(`ts_kst_min`, `HR_mean`) 중심  
  *(현재 코드에는 주차 라벨링/요약 예시가 일부 함께 포함될 수 있습니다.)*

### Steps
- 원시 step 세션 → 1분 단위 변환
- 연속 세션 구성/정규화
- 누적값 → 증분(step increment) 계산
- 기본 QC(이상치/급격 변동 탐지 유틸 포함)

### Survey
- 설문 원본 정리(필드 정리/결측 처리 관점)
- PHQ-9 / GAD-7 점수화 및 이분화 라벨 생성(Python)
- (참고) TS 기반 설문 분석 유틸(여러 설문 스코어링/해석 로직 포함)

---

## 2) 디렉토리 구조

```text
preprocessing/
├─ requirements.txt
├─ README.md
├─ notebooks/
│  ├─ ppg_data_visualization.ipynb
│  ├─ 전처리(ppg green).ipynb
│  └─ 전처리(response, heart rate, step count).ipynb
└─ src/
   ├─ __init__.py
   ├─ utils/
   │  ├─ __init__.py
   │  └─ week.py
   ├─ ppg/
   │  ├─ __init__.py
   │  ├─ io.py
   │  ├─ filtering.py
   │  ├─ sqi.py
   │  └─ preprocessing.py
   ├─ heart_rate/
   │  ├─ __init__.py
   │  └─ preprocessing.py
   ├─ steps/
   │  ├─ __init__.py
   │  └─ preprocessing.py
   └─ survey/
      ├─ __init__.py
      ├─ preprocessing.py
      └─ surveyAnalysis.ts
```

---

## 3) 설치 및 실행 환경

### Python 의존성 설치
```bash
pip install -r requirements.txt
```
- `requirements.txt` 포함 항목: `numpy`, `pandas`, `scipy`, `tqdm`

### TS 파일(`surveyAnalysis.ts`) 사용 시
- 본 폴더는 **Node 프로젝트로 패키징되어 있지 않습니다.**
- TS 파일은 **참고 코드/이식용** 성격으로 포함되어 있으며, 실제 사용 시에는 별도 TS/Node 프로젝트로 옮겨 빌드 환경을 구성하는 것을 권장합니다.

---

## 4) notebooks/ (원본 근거 자료)

### `notebooks/ppg_data_visualization.ipynb`
- PPG 데이터를 읽고(폴더 단위), **필터링/피크 검출/품질 확인**을 진행합니다.
- SQI(품질 지표) 계산 방식과 결과 확인(시각화 포함)에 사용된 노트북입니다.
- `src/ppg/io.py`, `src/ppg/filtering.py`, `src/ppg/sqi.py`의 근거가 되는 코드가 포함됩니다.

### `notebooks/전처리(ppg green).ipynb`
- PPG Green 중심 전처리 흐름(컬럼 정리, 시간 처리, 구간 구성, 품질 기준 등)을 정리한 노트북입니다.
- 주차 라벨링 등(필요 시)을 포함할 수 있으며, 일부 로직은 `src/utils/week.py`로 분리되어 있습니다.

### `notebooks/전처리(response, heart rate, step count).ipynb`
- 응답(설문), 심박, 걸음수 데이터 전처리 흐름이 함께 들어있는 노트북입니다.
- `src/heart_rate/preprocessing.py`, `src/steps/preprocessing.py`, `src/survey/preprocessing.py`, `src/survey/surveyAnalysis.ts`의 근거가 되는 코드가 포함됩니다.

---

## 5) src/ (모듈 상세)

`src/`는 전처리 코드를 **재사용 가능한 형태**로 모듈화한 묶음입니다.  
원본 노트북에서 발췌한 코드가 포함되어 있어, 일부 함수는 노트북에서 쓰던 전역 변수/경로를 그대로 가정할 수 있습니다(예: `BASE_DIR`, `UID`).  
레포 공유 목적이면 “로직 참고용”으로 충분하며, **재현 실행(즉시 실행 가능한 패키지)**이 목표라면 파일 경로/UID 등을 함수 인자로 받도록 인자화하는 정리를 권장합니다.

### 5.1 `src/ppg/`

#### `src/ppg/io.py`
- **목적**: PPG Green CSV를 “폴더 단위”로 읽어 하나의 DataFrame으로 결합합니다.
- **주요 함수**
  - `load_ppg_data_from_folder(folder_path, sensor_prefix="PpgGreen_")`
- **동작 요약**
  - `folder_path`에서 `sensor_prefix`로 시작하는 CSV 파일을 탐색
  - 각 파일을 `["time", "value"]` 컬럼으로 읽어 `concat`
- **입력/출력**
  - 입력: 폴더 경로(str), 파일 prefix(str)
  - 출력: 결합된 `pd.DataFrame` (없으면 `None`)
- **전제(스키마)**
  - 각 CSV가 `(time, value)` 2열 형태로 읽힐 수 있어야 합니다.

#### `src/ppg/filtering.py`
- **목적**: PPG 원신호에 band-pass 필터를 적용하는 유틸 함수입니다.
- **주요 함수**
  - `bandpass_filter(data, fs, lowcut=0.5, highcut=4.0, order=2)`
- **파라미터**
  - `data`: 1차원 신호 배열(`list`/`np.ndarray`)
  - `fs`: sampling rate(Hz)
  - `lowcut`, `highcut`: 통과 대역(Hz)
  - `order`: Butterworth 필터 차수

#### `src/ppg/sqi.py`
- **목적**: 1분 단위 윈도우 품질 평가에 사용하는 SQI(Signal Quality Index) 계산 로직입니다.  
  SQI는 “노이즈 비율”을 계산한 뒤 아래 형태로 산출합니다.
  - `SQI = 1 - final_noise`
- **주요 함수 구성**
  - `calculate_rr_noise(rr_intervals, config=None)`
    - RR 기반 노이즈 4종(A~D)을 계산하고 가중 평균으로 결합
    - A: 평균±kσ 밖 RR 비율 (`sigma_factor`, 기본 3.0)
    - B: 모드 대비 허용 범위 밖 RR 비율 (`hist_threshold`, 기본 ±15%)
    - C: Poincaré(χ²) 타원 밖 비율 (`ellipse_confidence`, 기본 0.90)
    - D: RR 변동 범위가 임계 초과인지 플래그 (`var_range_threshold`, 기본 2.0s)
    - 가중치 기본값: `weights=[2.0, 3.0, 1.0, 0.5]`
  - `calculate_ppg_var_diff_noise(ppg_signal, config=None)`
    - PPG 값 분산 기반(var) + 미분 기반(diff) 노이즈 비율 계산 후 결합
    - 기본 결합 가중: `var_diff_weights=[1.0, 5.0]`
  - `calculate_amp_peak_noise(ppg_signal, peak_count, config=None)`
    - 진폭 표준편차 범위/피크 개수 범위를 벗어나면 즉시 플래그로 품질 불량 판단
    - 기본값: `amp_std_min=100`, `amp_std_max=200000`, `peak_count_min=5`, `peak_count_max=200`
  - `calculate_noise(rr_intervals, ppg_signal, peak_count, config=None)`
    - RR 기반 노이즈 + PPG 기반 노이즈를 결합하여 `combined_noise` 산출
  - `calculate_SQI(rr_intervals, ppg_signal, peak_count, config=None)`
    - 최종 `sqi`, `final_noise`, `detail`(중간 계산 결과 dict) 반환
- **출력 형태(권장 사용)**
  - `sqi`: 0~1 (1에 가까울수록 품질 양호)
  - `final_noise`: 0~1 (1에 가까울수록 노이즈 큼)
  - `detail`: 구성 요소별 ratio/flag/중간값(디버깅 및 기준 조정에 유용)

#### `src/ppg/preprocessing.py`
- **목적**: PPG DataFrame을 전처리 파이프라인에서 쓰기 좋은 형태로 표준화합니다.
- **주요 함수**
  - `preprocess_ppg_data(df, time_col=None, value_col=None, dropna=True, sort=True, dedup=True)`
- **동작 요약**
  - 시간 컬럼/값 컬럼을 자동 탐지(필요 시 `time_col`, `value_col` 지정 가능)
  - 표준 컬럼 `time`, `data`로 정리
  - (옵션) 결측 제거, 시간 정렬, `time` 중복 제거
- **출력**
  - `time`, `data` 중심으로 정리된 `pd.DataFrame`

---

### 5.2 `src/heart_rate/`

#### `src/heart_rate/preprocessing.py`
- **목적**: 심박 CSV를 읽어 타임스탬프 정규화 후, 1분 단위 평균 심박을 만들기 위한 전처리 흐름을 포함합니다.
- **주요 함수**
  - `preprocess_watch_heart_rate()`
- **포함된 처리 유형(코드 기준)**
  - 원본 CSV 로딩(경로/UID 등 전역 설정을 사용하는 형태일 수 있음)
  - UTC epoch(ms) → KST datetime 변환
  - 1분 단위 집계(`ts_kst_min`, `HR_mean`)
  - (구현에 따라) 주차 라벨링/요약 로직이 일부 포함될 수 있음

---

### 5.3 `src/steps/`

#### `src/steps/preprocessing.py`
- **목적**: 걸음수 데이터(세션 기반/누적 기반)를 분 단위 데이터로 정규화하고, QC를 적용하기 위한 유틸/전처리 함수 모음입니다.
- **주요 함수/유틸(코드 기준)**
  - `preprocess_watch_step_count()`: 전체 전처리 진입점 성격(노트북 기반)
  - `create_minute_data(df, time_col="time", step_col="step")`: 세션 데이터를 1분 단위로 변환
  - `create_continuous_sessions(df, time_col="time", gap_threshold="5min")`: 시간 간격 기반 연속 세션 구성
  - `merge_zero_step_sessions(df, step_col="step")`: 0 step 구간 처리/병합
  - `normalize_continuous_sessions(df, time_col="time", step_col="step")`: 세션 정규화(정렬/중복 처리 포함 가능)
- **QC 함수(코드 기준)**
  - `is_reasonable_step_value(step, upper_limit=1000)`: 값 범위 기반 1차 QC
  - `detect_outliers_iqr(series, factor=1.5)`: IQR 기반 이상치 탐지
  - `detect_outliers_zscore(series, threshold=3.0)`: z-score 기반 이상치 탐지
  - `detect_sudden_jumps(series, threshold=200)`: 급격 변동 탐지
- **전제**
  - 입력 파일명/경로(예: `watch_step_count_sessions_normalized_minutes.csv`)는 노트북 내 가정이 남아 있을 수 있습니다.
  - 실행 안정성을 높이려면 (1) 입력 경로 인자화, (2) 컬럼 스키마 주석 명시를 권장합니다.

---

### 5.4 `src/survey/`

#### `src/survey/preprocessing.py`
- **목적**: 설문 응답을 정리하고, PHQ-9 / GAD-7 점수·라벨을 생성합니다.
- **주요 함수**
  - `preprocess_response(df)`: response(설문) 원본을 DataFrame으로 정리
  - `score_phq9(items, core_threshold=2, total_threshold=10)`
    - 입력: PHQ-9 9문항 점수(0~3) 리스트
    - 산출:
      - `phq9_total`(총점)
      - `phq9_severity`(중증도 범주)
      - `phq9_core_positive`(1/2번 핵심문항 양성 여부)
      - `phq9_binary`(이분형 라벨)
  - `score_gad7(items, threshold=6)`
    - 입력: GAD-7 7문항 점수(0~3) 리스트
    - 산출: `gad7_total`, `gad7_binary`
- **결측 처리**
  - `None`/`NaN`은 기본적으로 0점으로 처리하는 형태로 구현되어 있습니다.

#### `src/survey/surveyAnalysis.ts`
- **목적**: 프론트/TS 환경에서 설문 블록 단위로 점수 계산 및 해석 문구를 만드는 분석 유틸입니다.
- **핵심 엔트리**
  - `analyzeSurveyBlock(block: SurveyBlock): AnalysisResult`
- **지원 설문(코드 기준)**
  - PHQ-9, CES-D, GAD-7, ISI, STRESS, INQ, WHOQOL, S-SCALE, HAM-D 등
  - 일부는 “분석 보류” 형태로 처리될 수 있음(예: HAM-A, CNS-VS)
- **사용 맥락**
  - 본 폴더는 TS 빌드 환경이 없으므로, 실제로 사용하려면 해당 TS 파일을 사용하는 프로젝트로 옮겨 타입/의존성(`SurveyBlock` 등)을 맞추는 방식이 일반적입니다.

---

### 5.5 `src/utils/`

#### `src/utils/week.py`
- **목적**: 전처리 이후, 데이터를 주차(week) 단위로 묶기 위한 라벨링 유틸입니다.
- **주요 함수**
  - `load_week_cutoffs()`
    - 주차 경계 시점(cutoff)을 로딩하는 형태로 설계
  - `assign_week_label(ts_kst, cutoffs)`
    - KST timestamp가 어느 주차에 속하는지 반환
- **주차 라벨링 개념**
  - `cutoffs = [(1, T1), (2, T2), (3, T3), ...]`일 때:
    - `ts < T1` → `0`
    - `T1 ≤ ts < T2` → `1`
    - `T2 ≤ ts < T3` → `2`
    - `ts ≥ 마지막 cutoff` → 마지막 주차
- **참고**
  - 노트북 실행 전제를 일부 포함할 수 있어(경로/UID 등), 단독 실행 목적이면 `cutoffs`를 인자로 직접 주입하는 형태로 사용하는 편이 안정적입니다.
