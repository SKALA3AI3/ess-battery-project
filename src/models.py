"""Model factory functions."""

from __future__ import annotations

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import RANDOM_SEED


def build_elastic_net_pipeline(
    *,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Create the ElasticNet training pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=random_state,
                    max_iter=10000,
                ),
            ),
        ],
    )


def build_xgboost_regressor(
    *,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    subsample: float = 0.8,
    colsample_bytree: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> XGBRegressor:
    """Create the XGBoost regressor used in the project."""

    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=1,
    )

