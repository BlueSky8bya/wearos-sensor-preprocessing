# WearableBundle

Wear OS 기반 센서 데이터 수집 앱(`Wear_os_Sensor`)과 전처리 코드(`preprocessing`)를 함께 묶어 공유하는 저장소입니다.

---

## 1) 포함 범위

### Wear_os_Sensor (Wear OS 앱)
- Wear OS 워치에서 센서 데이터를 수집하는 앱 프로젝트
- (예) PPG/심박/가속도 등 센서 스트림을 파일(CSV 등)로 저장 또는 전송

### preprocessing (전처리 코드)
- 수집된 데이터(CSV/로그)를 분석에 투입 가능한 형태로 정규화/윈도우링/QC/라벨링
- 상세 내용은 `preprocessing/README.md` 참고

---

## 2) 디렉토리 구조

text
WearableBundle/
├─ README.md
├─ Wear_os_Sensor/
└─ preprocessing/
   ├─ README.md
   ├─ requirements.txt
   ├─ notebooks/
   └─ src/

---

## 3) 실행 환경

### Wear OS 앱 빌드/실행
- Android Studio (권장: 최신 버전)
- JDK (Android Studio 내장 JDK 사용 권장)
- 테스트 디바이스: Wear OS 워치 + (필요 시) 페어링된 스마트폰

### 전처리
- Python 3.9+ 권장 (연구 환경에 맞게 3.10/3.11도 무방)
- 주요 의존성: `numpy`, `pandas`, `scipy`, `tqdm`

설치:
```bash
pip install -r preprocessing/requirements.txt
```

---

## 4) 사용 방법(빠른 시작)

### A. Wear OS 앱
1. Android Studio에서 `Wear_os_Sensor/` 폴더를 프로젝트로 엽니다.
2. 워치(또는 에뮬레이터)를 연결한 뒤 **Run** 합니다.
3. 앱 내 수집 설정(센서/주기/저장 방식 등)에 따라 데이터가 생성됩니다.

> 앱 실행/수집 방식은 프로젝트 구현에 따라 다를 수 있으므로, `Wear_os_Sensor/` 내부 문서/코드를 기준으로 확인하세요.

### B. 전처리 코드
1. 의존성 설치
```bash
pip install -r preprocessing/requirements.txt
```
2. 전처리 모듈 확인: preprocessing/src/
3. 상세 사용법/함수 설명: preprocessing/README.md 참고
