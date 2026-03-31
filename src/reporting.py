"""Reporting helpers for ESS battery experiments."""

from __future__ import annotations

import pandas as pd

from config import TARGET_MAPE
from evaluation import RegressionMetrics


def build_regression_report(metrics: RegressionMetrics, *, target_mape: float = TARGET_MAPE) -> pd.DataFrame:
    """Build the project reporting table."""

    report_rows = [
        {"구분": "Train (Batch 1 CV)", "MAPE (%)": f"{metrics.train_cv_mape:.2f}", "비고": "Batch 1 내 5-Fold CV 평균 성능"},
        {"구분": "Valid (Batch 1 Hold-out)", "MAPE (%)": f"{metrics.valid_mape:.2f}", "비고": "Batch 1 내 Hold-out 검증 성능"},
        {"구분": "Test (Batch 2)", "MAPE (%)": f"{metrics.test2_mape:.2f}", "비고": "1차 테스트 셋 성능"},
        {"구분": "Gap (Train-Valid)", "MAPE (%)": f"{metrics.train_cv_mape - metrics.valid_mape:+.2f}", "비고": "(+) : 과적합 의심"},
        {"구분": "Gap (Valid-Test)", "MAPE (%)": f"{metrics.valid_mape - metrics.test2_mape:+.2f}", "비고": "(+) : 배치간 일반화 저하 의심"},
        {"구분": "Gap (Target-Test2)", "MAPE (%)": f"{target_mape - metrics.test2_mape:+.2f}", "비고": f"Target : 원논문 {target_mape}% (음수면 목표 미달)"},
        {"구분": "Test (Batch 3) - Add.", "MAPE (%)": f"{metrics.test3_mape:.2f}", "비고": "2차 테스트 셋 성능 (분포 상이)"},
        {"구분": "Gap (Batch2-Batch3)", "MAPE (%)": f"{metrics.test2_mape - metrics.test3_mape:+.2f}", "비고": "Test 성능 간 비교"},
        {"구분": "Gap (Target-Test3)", "MAPE (%)": f"{target_mape - metrics.test3_mape:+.2f}", "비고": "Batch 3 기준, 원논문 성능 비교"},
    ]
    return pd.DataFrame(report_rows)

