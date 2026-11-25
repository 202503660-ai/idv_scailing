# 스케일링 법칙은 실제 AI 모델에서도 성립했는가? (2019–2024)

2026학년도 2학기 한국외국어대학교 산업데이터시각화 개인 프로젝트 Streamlit 대시보드입니다.  
**Epoch AI** 공개 벤치마크 데이터(`benchmarks_with_model_accessibility.csv`)를 기반으로, 실제 공개·비공개 AI 모델의 시간 추이·접근성(Open/Closed) 분포·스케일링 법칙(단일/다중 회귀)·회귀 가정 진단(LINE·HC3)을 인터랙티브 차트로 검증합니다.

---

## 1. 데이터 출처 및 집계 기준

- **원자료 출처**: [Epoch AI · open-model-trends / `benchmarks_with_model_accessibility.csv`](https://github.com/epoch-research/open-model-trends/blob/main/data/benchmarks_with_model_accessibility.csv)
- **집계 기준**: `1행 = 1 AI 모델`
  - 원본 132행 중 비모델 비교 기준선 `'Random chance'` 1행(`MMLU=0.25`, `BBH=0.25`)을 분리해 차트 기준선으로 표기하고, 실제 AI 모델 **131종**(`2019-10-23 ~ 2024-09-12`)을 분석합니다.
  - 원본 24개 열 명칭·`0~1` 원척도·결측값·이상값을 임의 가공하거나 대체·삭제하지 않고, 각 회귀별 유효 표본 `n`을 투명하게 표시합니다.

---

## 2. 대시보드 구성 (6개 탭)

| 탭 | 주요 시각화 및 분석 내용 |
| --- | --- |
| **01 · 원자료** | 24개 컬럼 5대 역할 선버스트 계층 휠(클릭 시 `Open 유효 / Closed 유효 / 결측` 확대) · 컬럼 유형 분류 기준 토글 · 유효 관측률 막대(`50%` 기준선) · Open/Closed 공개율 비교 · 원자료 표 & CSV 다운로드 |
| **02 · 시간 추이** | 출시 시점별 벤치마크(`MMLU`/`BBH`) 및 규모(`Training compute (FLOP)`, `Active Parameters`, `Dataset size`) 추이 + 단일 OLS·95% CI/PI |
| **03 · Open / Closed** | 연도별 Open/Closed 모델 수(Bar) 및 전체 비율(Pie) + 분포 요약 |
| **04 · 스케일링 법칙** | 4대 규모 축(`Training compute (FLOP)`, `Active Parameters`, `Model size (parameters)`, `Dataset size`) vs `MMLU/BBH` 산점도 + 개별 OLS 비교표 |
| **05 · 다중 회귀** | `Model A (2변수: Active Parameters + Dataset size)` vs `Model B (3변수: + FLOP)` 실측값 vs 예측값 플롯 · 수정 $R^2$ · VIF 다중공선성 진단 |
| **06 · 회귀 진단** | 7개 모형 × 3종 잔차 플롯(`Residual vs Fitted`, `Normal Q-Q`, `Residuals over Date`) · LINE 4대 가정 검정(`RESET`, `DW`, `Shapiro`, `BP`) · `OLS vs HC3` 강건 표준오차 비교 · **핵심 결론 및 해석의 한계** |

> 모든 탭의 사이드바 위젯(`Benchmark`, `Date`, `Open/Closed`, `Plot Summary 표시`, `OLS + 95% CI`, `95% PI`, `HC3`)을 바꾸면 상단 KPI·차트·회귀 계수·요약 카드가 실시간으로 연동됩니다.

---

## 3. 실행 및 배포 방법

### 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud 배포
- **Main file path**: `app.py`
- **Advanced settings → Python version**: **`3.12`** 선택 권장 (`requirements.txt`의 `numpy==1.26.4`, `pandas==2.3.3`, `scipy==1.16.3` 사전 빌드 휠 호환)

---

## 4. 폴더 구조

```text
.
├── app.py                     # Streamlit 메인 앱 (6탭 레이아웃 · 사이드바 필터 · KPI)
├── charts.py                  # 순수 Plotly 차트 빌더 (Sunburst · Scatter · Bar · Pie · 진단 플롯)
├── scaling_stats.py           # OLS · 다중회귀 · 95% CI/PI · VIF · LINE 4대 가정 · HC3 계산 엔진
├── astryx_tokens.py           # 디자인 토큰 및 색맹 안전 데이터 팔레트
├── requirements.txt           # 고정 패키지 버전
├── .streamlit/config.toml     # Streamlit 라이트 테마 설정
├── data/
│   └── benchmarks_with_model_accessibility.csv
└── docs/
    ├── presentation_qa_guide.md
    └── verification_5cond.md
```
