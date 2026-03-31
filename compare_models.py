from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
RESULT_ROOT = PROJECT_ROOT / "result"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation import evaluate_model_suite
from models import get_model_builders
from preprocess import prepare_feature_table, prepare_regression_data, split_by_batch
from reporting import build_model_comparison_report, build_regression_report


def main() -> None:
    feature_table = prepare_feature_table()
    print(f"데이터 로딩 및 정제 완료 (총 {len(feature_table)}개 셀 확보)")

    selected_features = ["log_dq_variance", "dq_min", "mean_Tavg", "mean_QD"]
    print(f"사용 feature: {selected_features}")

    splits = split_by_batch(feature_table)
    regression_data = prepare_regression_data(splits, selected_features)

    model_results = evaluate_model_suite(get_model_builders(), regression_data)

    for model_name, experiment_result in model_results.items():
        print(f"\n[{model_name}]")
        print(build_regression_report(experiment_result.metrics).to_string(index=False))

    comparison_report = build_model_comparison_report(model_results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / "model_comparison.csv"
    comparison_report.to_csv(output_path, index=False)

    print("\n[Model Comparison Summary]")
    print(comparison_report.to_string(index=False))
    print(f"\n비교 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
