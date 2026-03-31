import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# CSV 파일 로드
df = pd.read_csv('feat_all.csv')

# NaN 값 제거 또는 채우기
df = df.dropna()  # NaN이 있는 행 제거

# 데이터 분할: Batch 1 = Train/Valid, Batch 2 = Test2, Batch 3 = Test3
batch1_data = df[df['batch'] == 'Batch 1']
test2_data = df[df['batch'] == 'Batch 2']
test3_data = df[df['batch'] == 'Batch 3']

# 타겟과 피처 분리 (특정 피처만 선택)
target = 'cycle_life'
selected_features = ['log_dq_variance', 'dq_min', 'mean_Tavg', 'mean_QD']

# Batch 1을 Train과 Valid로 분할
X_batch1 = batch1_data[selected_features]
y_batch1 = batch1_data[target]
X_train, X_valid, y_train, y_valid = train_test_split(X_batch1, y_batch1, test_size=0.2, random_state=42)

X_test2 = test2_data[selected_features]
y_test2 = test2_data[target]
X_test3 = test3_data[selected_features]
y_test3 = test3_data[target]

print(f"Selected features: {selected_features}")
print(f"Train data shape: {X_train.shape}")
print(f"Valid data shape: {X_valid.shape}")
print(f"Test2 data shape: {X_test2.shape}")
print(f"Test3 data shape: {X_test3.shape}")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error

def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# 모델 정의 (하이퍼파라미터 튜닝 추가, XGBoost와 ElasticNet 추가)
models = {
    'RandomForest': GridSearchCV(RandomForestRegressor(random_state=42), 
                                 {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}, cv=3),
    'LinearRegression': LinearRegression(),
    'Ridge': GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0]}, cv=3),
    'Lasso': GridSearchCV(Lasso(random_state=42), {'alpha': [0.01, 0.1, 1.0]}, cv=3),
    'XGBoost': GridSearchCV(XGBRegressor(random_state=42, objective='reg:squarederror'), 
                            {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 10]}, cv=3),
    'ElasticNet': GridSearchCV(ElasticNet(random_state=42), {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}, cv=3)
}

# 결과 저장
results = {}

for name, model in models.items():
    # CV on Train (Batch 1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_percentage_error')
    mape_cv = -cv_scores.mean() * 100  # MAPE (%)
    
    # Train model on full Train
    model.fit(X_train, y_train)
    
    # Predict on Valid (Batch 1 hold-out)
    y_pred_valid = model.predict(X_valid)
    mape_valid = calculate_mape(y_valid, y_pred_valid)
    
    # Predict on Test (Batch 2)
    y_pred_test2 = model.predict(X_test2)
    mape_test2 = calculate_mape(y_test2, y_pred_test2)
    
    # Predict on Test (Batch 3)
    y_pred_test3 = model.predict(X_test3)
    mape_test3 = calculate_mape(y_test3, y_pred_test3)
    
    results[name] = {
        'CV_Train': mape_cv,
        'Holdout_Valid': mape_valid,
        'Test2': mape_test2,
        'Test3': mape_test3,
        'Test2_Predictions': y_pred_test2,
        'Test3_Predictions': y_pred_test3
    }

# Target MAPE (원논문 9.1%)
target_mape = 9.1

# 각 모델별 테이블 출력
for name, res in results.items():
    print(f"### {name}")
    print()
    print("| 구분 |  | MAPE (%) | 비고 |")
    print("| --- | --- | --- | --- |")
    print(f"| {name} |  |  |  |")
    print(f"| Train (Batch 1 CV) |  | {res['CV_Train']:.2f} |  |")
    print(f"| Valid (Batch 1 Hold-out) |  | {res['Holdout_Valid']:.2f} |  |")
    print(f"| Test (Batch 2)  |  | {res['Test2']:.2f} |  |")
    gap_train_valid = res['CV_Train'] - res['Holdout_Valid']
    print(f"|  | Gap (Train-Valid)  | {gap_train_valid:+.2f} | (+) : 과적합 의심 |")
    gap_valid_test = res['Holdout_Valid'] - res['Test2']
    print(f"|  | Gap (Valid-Test) | {gap_valid_test:+.2f} | (+) : 배치간 일반화 저하 의심 |")
    gap_target_test2 = target_mape - res['Test2']
    print(f"|  | Gap (Target-Test) | {gap_target_test2:+.2f} | Target : 원논문 9.1%  |")
    print(f"| Test (Batch 3)  |  | {res['Test3']:.2f} |  |")
    gap_batch2_batch3 = res['Test2'] - res['Test3']
    print(f"|  | Gap (Batch2-Batch3)  | {gap_batch2_batch3:+.2f} | Test 성능 간 비교 |")
    gap_target_test3 = target_mape - res['Test3']
    print(f"|  | Gap (Target-Test) | {gap_target_test3:+.2f} | Batch 3 기준, 원논문 성능 비교  |")
    print()

# 시각화: 실제 vs 예측 (Test2와 Test3)
plt.figure(figsize=(18, 12))  # 6개 모델이므로 크기 조정
for i, (name, res) in enumerate(results.items()):
    # Test2
    plt.subplot(6, 2, 2*i+1)
    plt.scatter(y_test2, [res['Test2_Predictions'][j] for j in range(len(y_test2))], alpha=0.7)
    plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--')
    plt.xlabel('Actual Cycle Life')
    plt.ylabel('Predicted Cycle Life')
    plt.title(f'{name}: Test (Batch 2) - Actual vs Predicted')
    plt.grid(True)
    
    # Test3
    plt.subplot(6, 2, 2*i+2)
    plt.scatter(y_test3, [res['Test3_Predictions'][j] for j in range(len(y_test3))], alpha=0.7)
    plt.plot([y_test3.min(), y_test3.max()], [y_test3.min(), y_test3.max()], 'r--')
    plt.xlabel('Actual Cycle Life')
    plt.ylabel('Predicted Cycle Life')
    plt.title(f'{name}: Test (Batch 3) - Actual vs Predicted')
    plt.grid(True)

plt.tight_layout()
plt.savefig('model_comparison_selected_features.png', dpi=150, bbox_inches='tight')
# plt.show()  # 주석 처리해서 파일만 저장

# 바 그래프: Test(Batch 2)와 Test(Batch 3)의 MAPE 비교
model_names = list(results.keys())
test2_mapes = [res['Test2'] for res in results.values()]
test3_mapes = [res['Test3'] for res in results.values()]

x = np.arange(len(model_names))  # 모델 개수만큼 x 위치
width = 0.35  # 바 너비

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, test2_mapes, width, label='Test (Batch 2)', color='skyblue')
bars2 = ax.bar(x + width/2, test3_mapes, width, label='Test (Batch 3)', color='salmon')

# 라벨 추가
ax.set_xlabel('Models')
ax.set_ylabel('MAPE (%)')
ax.set_title('MAPE Comparison: Test (Batch 2) vs Test (Batch 3)')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()

# 바 위에 값 표시
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('mape_bar_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()  # 주석 처리해서 파일만 저장

print("\n모델 비교 완료. 결과가 'model_comparison_selected_features.png'와 'mape_bar_comparison.png'에 저장되었습니다.")