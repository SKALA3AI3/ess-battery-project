from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation import evaluate_log_target_model
from models import build_xgboost_regressor
from preprocess import prepare_feature_table, prepare_regression_data, split_by_batch
from reporting import build_regression_report


def main() -> None:
    feature_table = prepare_feature_table()
    print(f"데이터 로딩 및 정제 완료 (총 {len(feature_table)}개 셀 확보)")

    selected_features = ["log_dq_variance"]
    splits = split_by_batch(feature_table)
    regression_data = prepare_regression_data(splits, selected_features)

    model = build_xgboost_regressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=1.0,
    )
    result = evaluate_log_target_model(model, regression_data)
    report_df = build_regression_report(result.metrics)

    print("\n[Final Report - XGBoost]")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
