# 5조건 원자료 검산 표 (11주차 p.17 준수 기록)

> **대상 앱:** `my_deploy_scaling_law_final` (`app.py`, `charts.py`, `scaling_stats.py`)  
> **원자료 CSV:** `data/benchmarks_with_model_accessibility.csv` (`SHA-256: 63c6f27d1291...`, 133줄 = 헤더 1 + `Random chance` 1 + AI 모델 131종)  
> **독립 교차 검증 기준:** `pandas` + `scipy.stats.linregress` + `statsmodels.OLS` (최대 절대 오차 `5.55e-16`)  
> **사전 배포 점검 (`배포_전_점검.py`):** `EXIT_CODE=0` (`[고침] 0`)  

| 조건 | 필터 설정 | 표시 모델 수 | 성능 최고/중앙값 | FLOP 단순회귀 (`b1`, `R²`) | Model A / 경계 동작 | 검산 판정 |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| ① 기본 전체 | `MMLU`, `2019–2024`, `Open,Closed` | `131` | `max=0.887`, `med=0.671` (`n=98`) | `b1=+0.1268`, `R²=0.741` (`n=82`) | `Adj. R²=0.583`, `VIF=1.34` (`n=51`) | 일치 (`err ≤ 2.2e-16`) |
| ② 기간 축소 | `MMLU`, `2022–2024`, `Open,Closed` | `114` | `max=0.887`, `med=0.713` (`n=83`) | `b1=+0.1737`, `R²=0.768` (`n=67`) | `Adj. R²=0.725`, `VIF=1.02` (`n=37`) | 일치 (`err ≤ 2.2e-16`) |
| ③ 범주 전환 + 회귀 Off/On | `BBH`, `2019–2024`, `Closed` | `53` | `max=0.931`, `med=0.488` (`n=24`) | `b1=+0.0834`, `R²=0.721` (`n=20`) | `Adj. R²=0.752`, `VIF=1.34` (`n=17`) | 일치 (`err ≤ 3.3e-16`) |
| ④ 소표본·분산 0 경계 (2020–2020) | `MMLU`, `2020–2020`, `Open,Closed` | `4` | `max=0.397`, `med=0.237` (`n=4`) | `b1=+0.0621`, `R²=0.794` (`n=4`) | 생략 (`n=4 (< 6)`) / 시간회귀 `n=4 · 분산 0 (단일 값)` | 일치 (`err ≤ 5.6e-16`) |
| ④-b 소표본 경계 (2021–2021) | `MMLU`, `2021–2021`, `Open,Closed` | `3` | `max=0.600`, `med=0.600` (`n=1`) | 생략 (`n=1 (< 4)`) | 생략 (`n=0 (< 6)`) / 시간회귀 `n=1 (< 4)` | 일치 (`err ≤ 5.6e-16`) |
| ⑤ 빈 선택 (2019 Closed) | `MMLU`, `2019–2019`, `Closed` | `0` | `-` | `-` | `view.empty` 경고·정상 중지 | 일치 |

## 추가 자동 점검 통과 기록

- `AppTest` 0-lag 하향식 헤더 검증 (`sec1_scale_toggle`, `sec2_scale_toggle`): PASS
- `AppTest` `at.radio[3]` (`Residual vs Fitted` / `Normal Q-Q` / `Residuals over Date`) × 6개 모형 = 18/18: PASS
- `AppTest` `show_reg=False` 회귀 KPI·Model A/B·Section 4 조건부 비표시: PASS
- `Date=2020–2020` 분산 0 사유 분리(`n=4 (< 4)` 미출현) 및 단일 날짜 `Residuals over Date` 0예외: PASS
- `Closed` 단독 `VIF` 과학 표기(`(!)`) 및 단일 슬라이스 Pie 경계선 제거(`line.width=0`): PASS
- 전 필터 조합 산점도·진단 차트 점 잘림 `0건`, `[x_min, x_max]` 외삽 `0건`, `p = 0.000` `0건`, Bar `"0"` 라벨 `0건`: PASS
