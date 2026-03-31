### RandomForest

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| RandomForest |  |  |  |
| Train (Batch 1 CV) |  | 7.32 |  |
| Valid (Batch 1 Hold-out) |  | 4.94 |  |
| Test (Batch 2) |  | 40.02 |  |
|  | Gap (Train-Valid) | +2.38 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -35.08 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -30.92 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 13.82 |  |
|  | Gap (Batch2-Batch3) | +26.20 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -4.72 | Batch 3 기준, 원논문 성능 비교 |

### LinearRegression

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| LinearRegression |  |  |  |
| Train (Batch 1 CV) |  | 7.13 |  |
| Valid (Batch 1 Hold-out) |  | 6.49 |  |
| Test (Batch 2) |  | 55.52 |  |
|  | Gap (Train-Valid) | +0.64 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -49.04 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -46.42 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 13.53 |  |
|  | Gap (Batch2-Batch3) | +42.00 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -4.43 | Batch 3 기준, 원논문 성능 비교 |

### Ridge

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| Ridge |  |  |  |
| Train (Batch 1 CV) |  | 9.54 |  |
| Valid (Batch 1 Hold-out) |  | 7.19 |  |
| Test (Batch 2) |  | 36.35 |  |
|  | Gap (Train-Valid) | +2.35 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -29.16 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -27.25 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 14.01 |  |
|  | Gap (Batch2-Batch3) | +22.34 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -4.91 | Batch 3 기준, 원논문 성능 비교 |

### Lasso

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| Lasso |  |  |  |
| Train (Batch 1 CV) |  | 7.75 |  |
| Valid (Batch 1 Hold-out) |  | 7.05 |  |
| Test (Batch 2) |  | 53.36 |  |
|  | Gap (Train-Valid) | +0.70 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -46.31 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -44.26 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 12.90 |  |
|  | Gap (Batch2-Batch3) | +40.46 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -3.80 | Batch 3 기준, 원논문 성능 비교 |

### XGBoost

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| XGBoost |  |  |  |
| Train (Batch 1 CV) |  | 9.05 |  |
| Valid (Batch 1 Hold-out) |  | 7.06 |  |
| Test (Batch 2) |  | 55.15 |  |
|  | Gap (Train-Valid) | +1.99 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -48.09 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -46.05 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 16.16 |  |
|  | Gap (Batch2-Batch3) | +38.99 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -7.06 | Batch 3 기준, 원논문 성능 비교 |

### Elastic-Net

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| ElasticNet |  |  |  |
| Train (Batch 1 CV) |  | 9.14 |  |
| Valid (Batch 1 Hold-out) |  | 7.68 |  |
| Test (Batch 2) |  | 37.70 |  |
|  | Gap (Train-Valid) | +1.46 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | -30.02 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | -28.60 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 14.71 |  |
|  | Gap (Batch2-Batch3) | +22.99 | Test 성능 간 비교 |
|  | Gap (Target-Test) | -5.61 | Batch 3 기준, 원논문 성능 비교 |