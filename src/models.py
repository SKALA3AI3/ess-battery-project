"""Model factory functions."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import RANDOM_SEED


ModelBuilder = Callable[[], object]


def build_linear_regression() -> LinearRegression:
    """Create a plain linear regression model."""

    return LinearRegression()


def build_random_forest_grid_search(
    *,
    random_state: int = RANDOM_SEED,
    cv: int = 3,
) -> GridSearchCV:
    """Create the RandomForest GridSearchCV used in heeji.py."""

    estimator = RandomForestRegressor(random_state=random_state, n_jobs=1)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
    }
    return GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=1)


def build_ridge_grid_search(*, cv: int = 3) -> GridSearchCV:
    """Create the Ridge GridSearchCV used in heeji.py."""

    return GridSearchCV(
        estimator=Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=cv,
        n_jobs=1,
    )


def build_lasso_grid_search(
    *,
    random_state: int = RANDOM_SEED,
    cv: int = 3,
) -> GridSearchCV:
    """Create the Lasso GridSearchCV used in heeji.py."""

    return GridSearchCV(
        estimator=Lasso(random_state=random_state, max_iter=10000),
        param_grid={"alpha": [0.01, 0.1, 1.0]},
        cv=cv,
        n_jobs=1,
    )


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


def build_xgboost_grid_search(
    *,
    random_state: int = RANDOM_SEED,
    cv: int = 3,
) -> GridSearchCV:
    """Create the XGBoost GridSearchCV used in heeji.py."""

    estimator = XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=1,
    )
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
    }
    return GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=1)


def build_elastic_net_grid_search(
    *,
    random_state: int = RANDOM_SEED,
    cv: int = 3,
) -> GridSearchCV:
    """Create the ElasticNet GridSearchCV used in heeji.py."""

    estimator = ElasticNet(random_state=random_state, max_iter=10000)
    param_grid = {
        "alpha": [0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9],
    }
    return GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=1)


def get_model_builders() -> dict[str, ModelBuilder]:
    """Return the project-standard model registry."""

    return {
        "RandomForest": build_random_forest_grid_search,
        "LinearRegression": build_linear_regression,
        "Ridge": build_ridge_grid_search,
        "Lasso": build_lasso_grid_search,
        "XGBoost": build_xgboost_grid_search,
        "ElasticNet": build_elastic_net_grid_search,
    }
