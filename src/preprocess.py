"""Preprocessing utilities for ESS battery feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BAD_CELLS,
    DEFAULT_DATA_CANDIDATES,
    DEFAULT_TEST_SIZE,
    RANDOM_SEED,
    TARGET_LOG,
    TARGET_RAW,
)


@dataclass(frozen=True)
class DatasetSplits:
    """Container for batch-based dataset partitions."""

    full_frame: pd.DataFrame
    batch1_frame: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_batch2_frame: pd.DataFrame
    test_batch3_frame: pd.DataFrame


@dataclass(frozen=True)
class PreparedRegressionData:
    """Model-ready feature and target splits."""

    selected_features: list[str]
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test2: pd.DataFrame
    X_test3: pd.DataFrame
    y_train_log: pd.Series
    y_train_raw: pd.Series
    y_valid_raw: pd.Series
    y_test2_raw: pd.Series
    y_test3_raw: pd.Series
    imputation_values: pd.Series


def resolve_feature_table_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve the feature-table path with a sensible project-local fallback."""

    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Feature table not found: {path}")
        return path

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find feat_all.csv. Checked: "
        + ", ".join(str(candidate) for candidate in DEFAULT_DATA_CANDIDATES),
    )


def load_feature_table(feature_table_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw feature table."""

    return pd.read_csv(resolve_feature_table_path(feature_table_path))


def remove_known_bad_cells(
    frame: pd.DataFrame,
    bad_cells: Iterable[str] = BAD_CELLS,
) -> pd.DataFrame:
    """Remove the paper-documented bad Batch 1 cells."""

    return frame.loc[~frame["cell_id"].isin(list(bad_cells))].copy()


def drop_missing_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values."""

    return frame.dropna().copy()


def prepare_feature_table(
    feature_table_path: str | Path | None = None,
    *,
    drop_missing: bool = True,
    remove_bad_cells: bool = True,
) -> pd.DataFrame:
    """Apply the standard table-level preprocessing."""

    frame = load_feature_table(feature_table_path)
    if remove_bad_cells:
        frame = remove_known_bad_cells(frame)
    if drop_missing:
        frame = drop_missing_rows(frame)
    return frame.reset_index(drop=True)


def split_by_batch(
    frame: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> DatasetSplits:
    """Split the feature table into Batch 1 train/valid and Batch 2/3 tests."""

    batch1_frame = frame.loc[frame["batch"] == "Batch 1"].copy()
    batch2_frame = frame.loc[frame["batch"] == "Batch 2"].copy()
    batch3_frame = frame.loc[frame["batch"] == "Batch 3"].copy()

    train_frame, valid_frame = train_test_split(
        batch1_frame,
        test_size=test_size,
        random_state=random_state,
    )
    return DatasetSplits(
        full_frame=frame.copy(),
        batch1_frame=batch1_frame.reset_index(drop=True),
        train_frame=train_frame.reset_index(drop=True),
        valid_frame=valid_frame.reset_index(drop=True),
        test_batch2_frame=batch2_frame.reset_index(drop=True),
        test_batch3_frame=batch3_frame.reset_index(drop=True),
    )


def prepare_regression_data(
    splits: DatasetSplits,
    selected_features: list[str],
    *,
    target_log: str = TARGET_LOG,
    target_raw: str = TARGET_RAW,
) -> PreparedRegressionData:
    """Extract model-ready features/targets and apply train-median imputation."""

    X_train = splits.train_frame[selected_features].copy()
    X_valid = splits.valid_frame[selected_features].copy()
    X_test2 = splits.test_batch2_frame[selected_features].copy()
    X_test3 = splits.test_batch3_frame[selected_features].copy()

    imputation_values = X_train.median()
    X_train = X_train.fillna(imputation_values)
    X_valid = X_valid.fillna(imputation_values)
    X_test2 = X_test2.fillna(imputation_values)
    X_test3 = X_test3.fillna(imputation_values)

    return PreparedRegressionData(
        selected_features=selected_features,
        X_train=X_train,
        X_valid=X_valid,
        X_test2=X_test2,
        X_test3=X_test3,
        y_train_log=splits.train_frame[target_log].copy(),
        y_train_raw=splits.train_frame[target_raw].copy(),
        y_valid_raw=splits.valid_frame[target_raw].copy(),
        y_test2_raw=splits.test_batch2_frame[target_raw].copy(),
        y_test3_raw=splits.test_batch3_frame[target_raw].copy(),
        imputation_values=imputation_values,
    )

