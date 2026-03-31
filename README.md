# ESS 배터리 수명 예측 
Cycle 1 - 100까지의 데이터를 보고 최종 수명인 cycle life를 예측하는 목적입니다.
- 더 사용할 수 있는 배터리를 확인 가능합니다.
- 배터리의 남은 수명을 예측하여 친환경적인 배터리 사용을 할 수 있습니다.
- 배터리의 남은 수명을 예측하여 다른 회사에서 만든 배터리의 성능을 신뢰성 있게 볼 수 있는 지표가 될 수 있습니다.

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측)


## 파일 구조 (sample) 
```
├── data/
│   └── README.md          
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```


## 환경 설정 (sample) 
```bash
git clone https://github.com/SKALA3AI3/ess-battery-project
cd ess-battery-project
pip install -r requirements.txt
```


## EDA 

- Cycle Life 분포
	- 분포 형태 및 장단수명 비율 요약
        - [단수 : 장수]
        - Batch 1: 1:10
        - Batch 2: 28:3
        - Batch 3: 1:23
	- 핵심 발견 : Train으로 사용할 Batch 1 데이터의 경우 다른 데이터인 Batch 2와 다른 양상을 띕니다. 오히려 Batch 3와 유사한 양상을 띄며 장단수명 비율 또한 약 1:10으로 유사합니다. 반면에 Batch 3의 경우에는 단수명 배터리의 비율이 훨씬 높아 약 9:1의 장단수명 비율이 나타납니다.

- 열화 곡선 분석
	- 핵심 발견 : 
        - 비교적 단수명은 열화 속도가 가파르게 떨어집니다. 반면에 장수명의 경우 열화 속도가 완만하게 떨어집니다.
        - Knee Point란 어느 시점이 되면 배터리 수명이 뚝 떨어지는데 약 80% 이상일때 나타났습니다.
        - 하지만 Knee Point의 경우 200 cycle 이후에 보이므로 1-100 cycle 사이에서 나오지 않으므로 탐지하지 않아도 괜찮습니다.

- ΔQ(V) 곡선 분석
	- 핵심 발견 : 
        - 특정 전압 구간(2.8~3.2V)에서 급격한 변동이 나타납니다.
        - 이 변동성이 클수록 배터리 수명에 치명적인 영향을 미칩니다.

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
	- 핵심 발견 :
        - C-rate (충전 속도) 가 높을수록 수명이 짧은 경향임을 포착했습니다.
        - 이는 충전 프로토콜이 전체 수명에 많은 영향을 끼침을 알 수 있습니다.
        - 모델링 관점에서는 One Hot Encoding을 하려 하였지만 train하는 데이터의 양이 적었으므로 버렸습니다.

- (추가 확인한 내용 작성) 


## Modeling 

### 피처 엔지니어링 전략
EDA 결과를 바탕으로 선택한 피처와 그 근거를 기술


### 모델 선택 및 근거
- 후보 모델 : 
- 최종 모델 :
- 선택 이유 :


## 성능 결과
Format에 맞춰 작성


## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점
- 원인 가설 및 개선 방향


## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?


## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.


## 팀 구성
- 김영희 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch2)
- 박철수 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch3)