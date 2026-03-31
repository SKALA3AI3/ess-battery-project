"""Shared training and evaluation logic for log-target regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold

from config import CV_FOLDS, RANDOM_SEED
from preprocess import PreparedRegressionData


@dataclass(frozen=True)
class RegressionMetrics:
    """Summary metrics for one experiment."""

    train_cv_mape: float
    valid_mape: float
    test2_mape: float
    test3_mape: float


@dataclass(frozen=True)
class ExperimentResult:
    """Bundled training output."""

    fitted_model: object
    metrics: RegressionMetrics


def calculate_mape_percent(y_true, y_pred) -> float:
    """Return MAPE in percent."""

    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


def restore_cycle_life_from_log10(predictions_log):
    """Convert log10 cycle life predictions back to raw cycle life."""

    return np.power(10.0, predictions_log)


def cross_validate_log_target_model(
    model,
    data: PreparedRegressionData,
    *,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> float:
    """Run Batch 1 train-only cross-validation."""

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_mapes: list[float] = []

    for train_idx, valid_idx in kfold.split(data.X_train):
        fold_model = clone(model)
        X_fold_train = data.X_train.iloc[train_idx]
        y_fold_train_log = data.y_train_log.iloc[train_idx]
        X_fold_valid = data.X_train.iloc[valid_idx]
        y_fold_valid_raw = data.y_train_raw.iloc[valid_idx]

        fold_model.fit(X_fold_train, y_fold_train_log)
        predictions_raw = restore_cycle_life_from_log10(fold_model.predict(X_fold_valid))
        fold_mapes.append(calculate_mape_percent(y_fold_valid_raw, predictions_raw))

    return float(np.mean(fold_mapes))


def evaluate_log_target_model(model, data: PreparedRegressionData) -> ExperimentResult:
    """Fit once on Batch 1 train and evaluate on valid/test batches."""

    train_cv_mape = cross_validate_log_target_model(model, data)

    fitted_model = clone(model)
    fitted_model.fit(data.X_train, data.y_train_log)

    preds_valid_raw = restore_cycle_life_from_log10(fitted_model.predict(data.X_valid))
    preds_test2_raw = restore_cycle_life_from_log10(fitted_model.predict(data.X_test2))
    preds_test3_raw = restore_cycle_life_from_log10(fitted_model.predict(data.X_test3))

    metrics = RegressionMetrics(
        train_cv_mape=train_cv_mape,
        valid_mape=calculate_mape_percent(data.y_valid_raw, preds_valid_raw),
        test2_mape=calculate_mape_percent(data.y_test2_raw, preds_test2_raw),
        test3_mape=calculate_mape_percent(data.y_test3_raw, preds_test3_raw),
    )
    return ExperimentResult(fitted_model=fitted_model, metrics=metrics)

