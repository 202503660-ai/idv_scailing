# 발표 및 교수님 Q&A 정리 (최종 6탭 통합본 기준)

> **대상 앱:** `my_deploy_scaling_law_final` ([app.py](../app.py))
> **단일 핵심 질문:** **"스케일링 법칙은 실제 AI 모델에서도 성립했는가? (2019–2024)"**
> **단일 공식 CSV:** `data/benchmarks_with_model_accessibility.csv` (원본 133줄 = 헤더 1행 + 109번째 줄 `'Random chance' (0.25)` 기준선 1행 + **실제 AI 모델 131종**, `2019-10-23 ~ 2024-09-12`, 24개 원본 열 무번역 유지)

---

## 1. 회귀분석 방법론 (참고자료 6·7주차 기법 기준)

원 CSV의 `0~1` 원값 그대로 6·7주차 통계 절차(OLS, 95% CI/PI, F·t 검정, 수정 $R^2$, VIF, LINE 4대 가정, HC3)만 적용했습니다.

1. **전처리 및 $\log_{10}$ 변환 (7주차 · 9주차)**
   - 109번째 줄 `'Random chance'`(`MMLU=0.25`, `BBH=0.25`)만 분리해 실제 모델 **131종**으로 분석했고, `0.25`는 차트 점선 기준선(`Random chance (0.25)`)으로 배치했습니다.
   - `Model size (parameters)`(`n=64`)의 64종 전부가 `Active Parameters`(`n=89`)와 동일한 값을 가지며, `Active Parameters`에는 `GPT-4`, `Llama 3/3.1`, `Chinchilla 70B`, `Gopher 280B`, `Mixtral 8x22B`, `DBRX` 등 25종이 추가되어 있어 주 파라미터 열로 사용했습니다(Tab 4 비교표·Tab 6 진단에서는 `Model size (parameters)`도 함께 확인 가능).
   - 규모 변수(`Training compute (FLOP)`, `Active Parameters`, `Model size (parameters)`, `Dataset size`)는 자릿수 격차가 크므로 $\log_{10}$ 변환 후 OLS를 적합했습니다(`b1` = 규모가 10배 커질 때 벤치마크 점수의 평균 변화).
   - 결측값은 임의 대체 없이 각 회귀별 유효 `n`을 표기했고, `Durbin-Watson` 검정을 위해 출시일(`Date`) 오름차순으로 정렬했습니다.
2. **단일 선형회귀(Simple OLS) · 95% CI/PI · 외삽 금지 (6주차 · 7주차)**
   - 회귀선과 95% 신뢰구간(파란 띠, 평균 구간)·95% 예측구간(주황 띠, 개별 모델 구간)은 관측 범위 `[x_min, x_max]` 안에서만 표시합니다(미래 외삽 금지).
   - `p-value`는 `0.000` 대신 `p < 0.001`로 표기하며, `HC3` 활성화 시 표 통계량 열은 `z`(병렬 표는 `t / z`)로 표시됩니다.
3. **다중 선형회귀(Multiple OLS) · 수정 $R^2$ · VIF (7주차)**
   - **Model A (2변수):** `MMLU ~ log10(Active Parameters) + log10(Dataset size)` (`n=51`)
     - `F = 35.94 (p < 0.001)`, `R² = 0.600`, **`Adj. R² = 0.583`**, **`Max VIF = 1.34`** (양호)
     - 다른 변수가 같을 때 `Active Parameters` 10배당 `+0.0981 (t = 5.06, p < 0.001)`, `Dataset size` 10배당 `+0.1295 (t = 3.34, p = 0.002)`로 둘 다 유의.
   - **Model B (3변수):** `Model A + log10(Training compute (FLOP))` (`n=51`)
     - $C \approx 6ND \Rightarrow \log_{10} C \approx \log_{10} N + \log_{10} D + \log_{10} 6$ 구조 때문에 `Adj. R²`가 `0.583 → 0.576`으로 하락하고 **`VIF`가 `FLOP 80.51`, `Active Parameters 49.36`, `Dataset size 11.09`**로 급증하며 모든 계수가 `p > 0.13`으로 불안정해짐(`FLOP` 부호 `-0.0487` 반전).
4. **LINE 4대 가정 진단 + HC3 강건 표준오차 (7주차)**
   - `Training compute (FLOP) vs MMLU`(`n=82`, `b1=+0.1268/10x`, `R²=0.741`)에서 `BP p < 0.001`(등분산 경고)이 나오지만, **`HC3` 강건 표준오차**를 적용해도 `SE: 0.0084 → 0.0096`, `z = 13.25 (p < 0.001)`로 결론이 유지됩니다.
   - `RESET p < 0.001`은 MMLU가 0.25~1.00 구간에 갇히는 포화 특성 때문이며, `DW = 0.90`은 세대별 모델 출시가 시간순으로 군집하는 자기상관을 반영합니다.

---

## 2. 6개 탭 구성 및 발표 4분 시나리오

| 탭 | 메인 플롯 & 컨트롤 | 우측 `PLOT SUMMARY` (사이드바 ON/OFF) | 핵심 수치 (MMLU 기본) |
|---|---|---|---|
| **01 · 원자료** *(30초)* | 3버튼 전환(`컬럼 계층 휠 (Sunburst)` · `유효 관측률 (Bar)` · `Open / Closed 공개율 (Bar)`) + 원자료 테이블·CSV 다운로드 | 원자료 구조(24컬럼·5그룹·규모 유효율·다중회귀 교집합) / Open·Closed 공개율 비교 | 24컬럼 5역할 휠(클릭 시 Open/Closed/결측 확대), 벤치마크 50%↑는 `MMLU 74.8% · BBH 54.2%` 2종뿐, `Active Params` 공개율 `Open 85.9% vs Closed 41.5%` |
| **02 · 시간 추이** *(45초)* | 1개 카드 통합 5버튼 (`MMLU/BBH` · `FLOP` · `Active Parameters` · `Model size (parameters)` · `Dataset size`) | 선택한 지표의 분포(IQR·중앙값)·연간 추세·$R^2$ | `MMLU` `+0.098/yr` (`R²=0.513`, `n=98`), `FLOP` 연 `5.06배` (`R²=0.588`, `n=94`), `Dataset` 연 `1.96배` (`R²=0.789`, `n=56`) |
| **03 · Open / Closed** *(25초)* | 연도별 모델 수(Bar) / 전체 비율(Pie) | `OPEN / CLOSED` 전용 분포(전체 수·기간·최다 출시 연도) | 총 `131종`: `Open 78 (60%)` vs `Closed 53 (40%)`, 최다 출시 `2024 (49개)` |
| **04 · 스케일링 법칙** *(55초)* | `FLOP` · `Active Parameters` · `Model size (parameters)` · `Dataset size` 4축 vs `MMLU/BBH` 산점도 + **4종 개별 OLS 비교표** | 선택한 축의 유효 쌍 표본·기울기(10x)·`p`·$R^2$/RSE | `FLOP` **`R²=0.741`**(`n=82`), `Active Params` `0.504`(`n=77`), `Model size` `0.495`(`n=55`), `Dataset` `0.386`(`n=51`) |
| **05 · 다중 회귀** *(45초)* | `Model A (2변수)` ↔ `Model B (3변수)` **실측값 vs 예측값($y$ vs $\hat{y}$)** 플롯 + 비교 배지 + 선택 모형 계수·VIF 표 | `MULTI OLS` 전용 지표(`n`, 변수 수, $R^2$, 수정 $R^2$, RSE, $F$, **최대 VIF 배지**) | `Model A` `Adj. R²=0.583 · VIF 1.34` vs `Model B` `Adj. R²=0.576 · VIF 80.51 (!)` (`n=51`) |
| **06 · 회귀 진단** *(40초)* | 7개 모형 × 3종 진단(`Residual vs Fitted` · `Normal Q-Q` · `Residuals over Date`) + `LINE`/`OLS vs HC3` 표 | 활성 진단 플롯에 맞춘 검정 지표(`RESET/BP` · `Shapiro` · `DW`) + `LINE 통과` 배지 | `FLOP vs MMLU`: `BP p < 0.001`이지만 `HC3 z=13.25 (p < 0.001)` 유지 |

---

## 3. 교수님 1분 Q&A 핵심 방어 포인트

1. **"이 대시보드가 Kaplan(2020) 스케일링 법칙을 증명한 건가?"** → 엄밀한 인과적 '증명'이 아니라 실제 공개 모델 벤치마크에서의 **실증 점검**입니다. Kaplan(2020)은 통제 실험에서 cross-entropy loss의 멱법칙을 검증했고, 우리 데이터는 기업·아키텍처·사후학습이 섞인 관측 자료의 다운스트림 정답률(`0~1`)에 로그-선형 OLS를 적용한 것입니다.
2. **"CSV가 133줄인데 왜 131종인가?"** → 1행은 열 이름 헤더, 109행은 비모델 비교 기준선 `'Random chance' (0.25)` 1행이라 분리해 차트 점선으로 표시했습니다. 실제 모델 누락은 0건입니다.
3. **"Total Parameter(`Model size (parameters)`)는 왜 따로 안 썼나?"** → `Model size (parameters)` 64종 전부가 `Active Parameters`(89종)와 100% 동일한 값이며, `GPT-4`·`Llama 3/3.1`·`Chinchilla`·`Mixtral` 등 25종은 `Active Parameters`에만 기록되어 있습니다. Tab 4 비교표와 Tab 6 진단에서 `Model size (parameters)`(`n=55`, `R²=0.495`)도 함께 대조할 수 있습니다.
4. **"Model B에 FLOP을 넣으니 왜 p=0.68, 부호가 음수(-0.0487)로 바뀌나?"** → $C \approx 6ND$ 관계로 인해 `VIF = 80.51` 다중공선성이 발생해 개별 계수가 불안정해지기 때문입니다(7주차 p.11–12).
5. **"BP 검정이 `p < 0.001`인데 회귀 결과를 믿을 수 있나?"** → 7주차 p.21 방식대로 `HC3` 강건 표준오차를 적용하면 `SE: 0.0084 → 0.0096`, `z = 13.25 (p < 0.001)`로 방향과 유의성이 그대로 유지됩니다.
6. **"호버에 기관(Organization)이 없는 이유는?"** → 원 CSV 24열에 기관 컬럼이 없어 임의 결합이나 가공 대신 원본 `Open/Closed` 공개 유형을 그대로 표기했습니다.
7. **"차트에 이름 태그가 붙은 모델은 어떤 기준으로 골랐나?"** → 수동으로 임의 선정하지 않고, 현재 필터 표본에서 **연도(`_year`)별 Y축 지표 실제 1위 모델**만 코드가 자동으로 추출해 표기합니다(동점 시 `MMLU` → `FLOP` 순).
8. **"왜 24개 컬럼 중 MMLU·BBH만 벤치마크로 골랐나?"** → Tab 1(`01 · 원자료`) 유효 관측률 차트에서 확인되듯 `MMLU(74.8%)`와 `BBH(54.2%)`만 50%를 넘으며, 나머지(`GPQA 34.4%`, `LMSys Elo 34.4%`, `SEAL 9~10%`, `HELM 6.9%`)는 결측이 과다해 규모 축과 교차하면 회귀 표본이 급감합니다.

---

## 4. 제출 전 최종 체크리스트 (개인프로젝트 안내 기준)

- [ ] GitHub 공개 저장소 업로드 및 Streamlit Community Cloud 배포 (**Advanced settings → Python 3.12 선택**, `numpy==1.26.4` 호환)
- [ ] 배포 URL에서 한글·차트·위젯·모바일(`390px`)·빈 선택 경고 직접 확인
- [ ] 발표자료 PDF 첫 장에 **배포 URL** 기재 + URL 장애 대비 **핵심 캡처** 삽입
- [ ] 발표자료 PDF 마지막 장에 **AI 사용 내역 1~2줄** 기재
- [ ] 최종 제출 3종(`① 최종 발표자료 PDF`, `② 배포 URL`, `③ GitHub 저장소 URL`) 제출 및 성적 확정 시까지 공개 유지
